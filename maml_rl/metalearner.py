import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import math
import ipdb


class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, exp_policy, baseline, exp_baseline, reward_net, reward_net_outer,
                 embed_size=100, gamma=0.95, fast_lr=0.5, tau=1.0, lr=7e-4, eps=1e-5, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.exp_policy = exp_policy
        self.baseline = baseline
        self.exp_baseline = exp_baseline
        self.reward_net = reward_net
        self.reward_net_outer = reward_net_outer
        self.embed_size = embed_size
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        
        self.eps_r = eps
        self.eps_z = eps
        self.eps_p = eps
        self.eps_e = eps
        self.eps_ro = eps

        self.lr_r = lr*10
        self.lr_z = lr*0.1
        self.lr_p = lr
        self.lr_e = lr*10
        self.lr_ro = lr

        self.alpha = 5e-8
        self.clip = 0.5

        self.z_old = nn.Parameter(torch.zeros((1,embed_size)))
        nn.init.xavier_uniform_(self.z_old)

        self.to(device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr_r, eps=self.eps_r)
        self.z_optimizer = optim.Adam([self.z_old], lr=self.lr_z, eps=self.eps_z)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_p, eps=self.eps_p)
        self.exp_optimizer = optim.Adam(self.exp_policy.parameters(), lr=self.lr_e, eps=self.eps_e)
        self.reward_outer_optimizer = optim.Adam(self.reward_net_outer.parameters(), lr=self.lr_ro, eps=self.eps_ro)
        
        self.iter = 0
        self.dice_wts = []
        self.dice_wts_detached = []
        self.exp_entropy = []

    def compute_reward_loss(self, rewards, rewards_pred):
        # TODO: smooth l1 loss ? 
        return (rewards - rewards_pred)**2

    def inner_loss(self, episodes, exp_update='dice', params=None):
        # inner loop update 
        states, actions, rewards = episodes.observations, episodes.actions, episodes.rewards
        rewards_pred = self.reward_net(states,actions,self.z).squeeze()
        loss = self.compute_reward_loss(rewards, rewards_pred)
        
        exp_pi = self.exp_policy(states, self.z.detach())
        self.exp_entropy.append(exp_pi.entropy().sum(dim=2))

        # Think of better objectives (predicting the reward function, hypothesis testing etc. which are denser)
        # Sparse rewards don't make sense for GD especially likelihood based GD. 
        # why even max^m the llhood wrt to actions taken by exploration agent.
        # Also log_probs of unlikely actions would explode (esp given the off policy setting)
        # also the importance weights would have very high variance. 
        # Hence need clipping etc. at least as in PPO etc. 

        if exp_update=='dice':
            exp_log_probs_diff = exp_pi.log_prob(episodes.actions)

            # TODO: This might be high variance so reconsider it later maybe.
            dice_wt = torch.exp(exp_log_probs_diff.sum(dim=2) - exp_log_probs_diff.sum(dim=2).detach())  
            self.dice_wts.append(dice_wt)
            # cum_wts = torch.exp(torch.log(self.dice_wts[-1]).cumsum(dim=0))
            # loss *= cum_wts
            self.dice_wts_detached.append(dice_wt.detach())
            self.dice_wts_detached[-1].requires_grad_()
            loss *= self.dice_wts_detached[-1]

        # if loss.dim() > 2:
        #     loss = torch.sum(loss, dim=2)
        #     exp_log_probs_diff = torch.sum(exp_log_probs_diff, dim=2)
        #     exp_log_probs_non_diff = torch.sum(exp_log_probs_non_diff, dim=2)

        # TODO: Do we need importance sampling?
        wts = episodes.mask
        loss = weighted_mean(loss, dim=0, weights=wts)
        loss = loss.mean()
        return loss

    def adapt(self, episodes, first_order=False, exp_update='dice'):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        # self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, exp_update)
        # Get the new parameters after a one-step gradient update
        curr_params, updated_params = self.update_z(loss, step_size=self.fast_lr, first_order=first_order)
        return curr_params, updated_params, loss

    def update_z(self, loss, step_size=0.5, first_order=False):
        grads = torch.autograd.grad(loss, self.z, create_graph=not first_order)
        updated_params = OrderedDict()
        curr_params = OrderedDict()
        updated_params['z'] = self.z - step_size * grads[0]
        curr_params['z'] = self.z

        return curr_params, updated_params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for i, task in enumerate(tasks):
            
            self.sampler.reset_task(task)
            # exploration policy episodes
            curr_params = OrderedDict()
            curr_params['z'] = self.z

            # no grad set in sample 
            train_episodes = self.sampler.sample(self.exp_policy, task, params=curr_params,
                gamma=self.gamma, device=self.device)

            curr_params, updated_params, _ = self.adapt(train_episodes, first_order=first_order)

            # exploitation policy episodes
            valid_episodes = self.sampler.sample(self.policy, task, params=updated_params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params, _ = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, updated_params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def surrogate_loss_rewardnet(self, episodes, old_pis=None, exp_update='dice'):
        losses, kls, pis, reward_losses, reward_losses_before = [], [], [], [], []
        assert old_pis is None, 'old pis should be None'
        old_pis = [None] * len(episodes)

        self.dice_wts = []
        self.dice_wts_detached = []
        self.exp_entropy = []
        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params, _ = self.adapt(train_episodes)
            self.baseline.fit(valid_episodes)

            with torch.no_grad():
                states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
                rewards_pred = self.reward_net_outer(states, actions, curr_params['z']).squeeze()
                reward_loss = self.compute_reward_loss(rewards, rewards_pred)
                reward_losses_before.append(reward_loss.mean())

            # Reward Objective
            rewards_pred = self.reward_net_outer(states, actions, updated_params['z'].detach()).squeeze()
            reward_loss = self.compute_reward_loss(rewards, rewards_pred)
            reward_losses.append(reward_loss.mean())

            # Policy Objective
            pi = self.policy(valid_episodes.observations, updated_params['z'])
            pis.append(detach_distribution(pi))
            old_pi = detach_distribution(pi)
            
            log_ratio = pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions)
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages, weights=valid_episodes.mask)
            
            loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
            losses.append(loss)

            # not being used 
            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), 
                torch.mean(torch.stack(reward_losses,dim=0)),
                torch.mean(torch.stack(reward_losses_before,dim=0)), 
                pis
                )

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, reward_loss_after, reward_loss_before, old_pis = self.surrogate_loss_rewardnet(episodes,
                                                                                     exp_update='dice')

        # reward_loss_after, reward_loss_before= self.test_func(episodes)
        
        grad_vals = self.gradient_descent_update(old_loss*10, reward_loss_after*1, episodes)
        return ((reward_loss_before, reward_loss_after)), old_loss, grad_vals

    def gradient_descent_update(self, old_loss, reward_loss, episodes):
        self.z_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.exp_optimizer.zero_grad()
        self.reward_outer_optimizer.zero_grad()

        self.iter += 1
        # why 5 ? 
        wts = math.exp(-self.iter/5)
        dice_grad = torch.autograd.grad(old_loss, self.dice_wts_detached, retain_graph=True)
        dice_wts_grad = []

        for i, (train_episodes, valid_episodes) in enumerate(episodes):
            dice_grad_normalized = weighted_normalize(dice_grad[i]).detach()
            returns = - dice_grad_normalized 

            self.exp_baseline.fit(train_episodes, returns)
            values = self.exp_baseline(train_episodes)
            
            advantages = returns - values.squeeze()
            ratio = self.dice_wts[i]
            action_loss = -ratio*advantages
            dice_wts_grad.append(action_loss)

        dice_grad_mean = 0
        for i in range(len(self.dice_wts)):
            dice_grad_mean += dice_wts_grad[i].sum() 
            
        scale = torch.sum(torch.tensor([dice_grad[i].sum() for i in range(len(dice_grad))]))/dice_grad_mean.sum().item()
        dice_grad_mean*=scale.detach().item()
        dice_grad_mean.sum().backward()

        (old_loss+wts*reward_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),self.clip)
        nn.utils.clip_grad_norm_(self.reward_net.parameters(),self.clip)
        nn.utils.clip_grad_norm_([self.z_old],self.clip)
        nn.utils.clip_grad_norm_(self.exp_policy.parameters(),self.clip)
        nn.utils.clip_grad_norm_(self.reward_net_outer.parameters(),self.clip)

        grad_vals = [self.z_old.grad.abs().mean().item(), 
                     self.policy.layer_pre1.weight.grad.abs().mean().item(),
                     self.exp_policy.layer_pre1.weight.grad.abs().mean().item(),
                     self.reward_net.layer_pre1.weight.grad.abs().mean().item(),
                     self.reward_net_outer.layer_pre1.weight.grad.abs().mean().item()]
        print("z_grad {:.9f}".format(grad_vals[0]))
        print("policy_grad {:.9f}".format(grad_vals[1]))
        print("exp_policy_grad {:.9f}".format(grad_vals[2]))
        print("reward_grad {:.9f}".format(grad_vals[3]))
        print("reward_grad_outer {:.9f}".format(grad_vals[4]))
        self.z_optimizer.step()
        self.reward_optimizer.step()
        self.reward_outer_optimizer.step()
        self.policy_optimizer.step()
        self.exp_optimizer.step()
        return grad_vals

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.exp_policy.to(device, **kwargs)
        self.exp_baseline.to(device, **kwargs)
        self.reward_net.to(device, **kwargs)
        self.reward_net_outer.to(device, **kwargs)
        self.z = self.z_old.to(device, **kwargs)
        self.device = device

    # def update_exp_policy_dice(self,old_loss, grads):
    #     self.exp_optimizer.zero_grad()
    #     for param, gradient in zip(self.exp_policy.parameters(),grads):
    #         param.grad = gradient
    #     self.exp_optimizer.step()

    # def get_returns(self, rewards):
    #     return_ = torch.zeros(rewards.shape[1]).to(self.device)
    #     returns = torch.zeros(rewards.shape[:2]).to(self.device)
    #     for i in range(len(rewards) - 1, -1, -1):
    #         return_ = self.gamma * return_ + rewards[i].detach()
    #         returns[i] = return_
    #     return returns

    # def gae(self, values, rewards, tau=1.0):
    #     values = values.squeeze(2).detach()
    #     # Add an additional 0 at the end of values for the estimation at the end of the episode
    #     values = F.pad(values, (0, 0, 0, 1))

    #     deltas = rewards + self.gamma * values[1:] - values[:-1]
    #     advantages = torch.zeros_like(deltas).float()
    #     gae = torch.zeros_like(deltas[0]).float()
    #     for i in range(len(rewards) - 1, -1, -1):
    #         gae = gae * self.gamma * tau + deltas[i]
    #         advantages[i] = gae
    #     returns = values[:-1] + advantages
    #     return advantages, returns


    # def hessian_vector_product(self, episodes, damping=1e-2):
    #     """Hessian-vector product, based on the Perlmutter method."""
    #     def _product(vector):
    #         kl = self.kl_divergence(episodes)
    #         grads = torch.autograd.grad(kl, self.policy.parameters(),
    #             create_graph=True)
    #         flat_grad_kl = parameters_to_vector(grads)

    #         grad_kl_v = torch.dot(flat_grad_kl, vector)
    #         grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
    #         flat_grad2_kl = parameters_to_vector(grad2s)

    #         return flat_grad2_kl + damping * vector
    #     return _product

    # def surrogate_loss(self, episodes, old_pis=None, exp_update='dice'):
    #     losses, kls, pis = [], [], []
    #     assert old_pis is None, 'old_pis should be None'
    #     old_pis = [None] * len(episodes)    

    #     for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
    #         curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
    #         self.baseline.fit(valid_episodes)
            
    #         pi = self.policy(valid_episodes.observations,updated_params['z']) #, params=updated_params)
    #         pis.append(detach_distribution(pi))

    #         if old_pi is None:
    #             old_pi = detach_distribution(pi)

    #         values = self.baseline(valid_episodes)
    #         advantages = valid_episodes.gae(values, tau=self.tau)
    #         advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

    #         log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
    #         if log_ratio.dim() > 2:
    #             log_ratio = torch.sum(log_ratio, dim=2)
    #         ratio = torch.exp(log_ratio)

    #         loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
    #         losses.append(loss)

    #         mask = valid_episodes.mask
    #         if valid_episodes.actions.dim() > 2:
    #             mask = mask.unsqueeze(2)
    #         kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
    #         kls.append(kl)

    #     return (torch.mean(torch.stack(losses, dim=0)),
    #             torch.mean(torch.stack(kls, dim=0)), pis)

    # def test_func(self, episodes):
    #     reward_losses, reward_losses_before = [], []
    #     for (train_episodes, valid_episodes) in episodes:
    #         # reward_loss_before=torch.Tensor(0)
    #         curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
    #         states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
    #         # updated_params = OrderedDict()
    #         # updated_params['z'] = torch.ones_like(self.z)
    #         # updated_params['z'][:self.embed_size//2]*=float(train_episodes.task[0])
    #         # updated_params['z'][self.embed_size//2:]*=float(train_episodes.task[1])
    #         rewards_pred = self.reward_net(states,actions,updated_params['z']).squeeze()
    #         reward_loss = self.compute_reward_loss(rewards, rewards_pred)
    #         reward_losses.append(reward_loss.mean())
    #         reward_losses_before.append(reward_loss_before)
    #     return (torch.mean(torch.stack(reward_losses,dim=0)),
    #             torch.mean(torch.stack(reward_losses_before,dim=0)))