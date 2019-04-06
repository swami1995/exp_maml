import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim

from collections import OrderedDict
import math

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
    def __init__(self, sampler, policy, exp_policy, baseline, exp_baseline, reward_net, num_updates=3, embed_size=100, gamma=0.95,
                 fast_lr=0.5, tau=1.0, lr=7e-4, eps=1e-5, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.exp_policy = exp_policy
        self.baseline = baseline
        self.exp_baseline = exp_baseline
        self.reward_net = reward_net
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.lr_r = lr
        self.eps_r = eps
        self.lr_z = lr*0.1
        self.eps_z = eps
        self.lr_p = lr
        self.eps_p = eps
        self.lr_e = lr*0.10
        self.eps_e = eps
        self.embed_size = embed_size
        self.value_coeff = 0.5
        self.entropy_coeff = 0
        self.clamp_param = 0.2
        self.num_updates = num_updates
        # pdb.set_trace()
        self.z_old = nn.Parameter(torch.zeros((1,embed_size)))
        self.z_opt = torch.zeros((4,1,embed_size)).to(device)
        for i in range(4):
            self.z_opt[i,0,i*embed_size//4:(i+1)*embed_size//4] = 1
        nn.init.xavier_uniform_(self.z_old)

        self.to(device)
        self.z_optimizer = optim.Adam([self.z_old], lr=self.lr_z, eps=self.eps_z)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr_r, eps=self.eps_r)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_p, eps=self.eps_p)
        self.exp_optimizer = optim.Adam(self.exp_policy.parameters(), lr=self.lr_e, eps=self.eps_e)
        self.check=False       
        self.iter = 0
        self.dice_wts = []
        self.clip = 0.5


    def inner_loss(self, episodes, current_params, exp_update='dice', params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        # pdb.set_trace()
        states, actions, rewards = episodes.observations, episodes.actions, episodes.rewards
        rewards_pred = self.reward_net(states,actions,current_params['z']).squeeze()
        loss = (rewards - rewards_pred)**2
        exp_pi, exp_value = self.exp_policy(states, current_params['z'].detach()) #### TODO: Should this be detached?
        exp_log_probs_non_diff = episodes.action_probs
        exp_log_probs_diff = torch.zeros_like(exp_log_probs_non_diff)
                                                        #### TODO: Think of better objectives (predicting the reward function, hypothesis testing etc. which are denser)
                                                        ####       Sparse rewards don't make sense for GD especially likelihood based GD. 
                                                        ####       why even max^m the llhood wrt to actions taken by exploration agent.
                                                        ####       Also log_probs of unlikely actions would explode (esp given the off policy setting)
                                                        ####       also the importance weights would have very high variance. 
                                                        ####       Hence need clipping etc. at least as in PPO etc. 
        if exp_update=='dice':
            # ipdb.set_trace()
            exp_log_probs_diff = exp_pi.log_prob(episodes.actions)
            self.dice_wts.append(torch.exp(exp_log_probs_diff.sum(dim=2) - exp_log_probs_non_diff.sum(dim=2)))  #### TODO: This might be high variance so reconsider it later maybe.
            cum_wts = torch.exp(torch.log(self.dice_wts[-1]).cumsum(dim=0))
            # self.dice_wts_copy = tor
            loss *= cum_wts

        if self.check:
            self.check=False

        # if loss.dim() > 2:
        #     loss = torch.sum(loss, dim=2)
        #     exp_log_probs_diff = torch.sum(exp_log_probs_diff, dim=2)
        #     exp_log_probs_non_diff = torch.sum(exp_log_probs_non_diff, dim=2)

        # wts = episodes.mask* torch.exp(log_probs.detach()-exp_log_probs_non_diff)     #### TODO: Do we need importance sampling?
        # loss = weighted_mean(loss * advantages, dim=0,
            # weights=wts)
        # loss = loss.reshape(-1)
        # loss = torch.topk(loss, loss.size(0)//10)
        loss = loss.mean()#.sum()/(loss>0).sum().float().detach()
        return loss

    def adapt(self, episodes, current_params, first_order=False, exp_update='dice'):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        # self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, current_params, exp_update)
        # Get the new parameters after a one-step gradient update
        # pdb.set_trace()
        curr_params, updated_params = self.update_z(loss, current_params, step_size=self.fast_lr, first_order=first_order)

        return curr_params, updated_params, loss

    def update_z(self, loss, curr_params, step_size=0.5, first_order=False):
        grads = torch.autograd.grad(loss, curr_params['z'],
            create_graph=not first_order)
        # curr_params = OrderedDict()
        # curr_params['z'] = updated_params['z']
        updated_params = OrderedDict()
        updated_params['z'] = curr_params['z'] - step_size * grads[0]

        return curr_params, updated_params


    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            
            self.sampler.reset_task(task)
            updated_params = []
            updated_param = OrderedDict()
            updated_param['z'] = self.z
            updated_params.append(updated_param)
            train_episodes = []
            for t in range(self.num_updates):
                # train_episodes.append(self.sampler.sample(self.exp_policy, task, params=updated_params[-1], gamma=self.gamma, device=self.device))
                train_episodes.append(self.sampler.sample(self.exp_policy, task, params=self.z_opt, gamma=self.gamma, device=self.device))                
                curr_param, updated_param, _ = self.adapt(train_episodes[-1], updated_params[-1], first_order=first_order)
                updated_params.append(updated_param)

            # valid_episodes = self.sampler.sample(self.policy, task, params=updated_params[-1],gamma=self.gamma, device=self.device)
            valid_episodes = self.sampler.sample(self.policy, task, params=self.z_opt, gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes, updated_params))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params, _ = self.adapt(train_episodes)
            pi, value = self.policy(valid_episodes.observations, updated_params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None, exp_update='dice'):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
            self.baseline.fit(valid_episodes)
            # old_pi = curr_params
            with torch.set_grad_enabled(old_pi is None):
                pi, value = self.policy(valid_episodes.observations,updated_params['z'])#, params=updated_params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def surrogate_loss_rewardnet(self, episodes, old_pis=None, exp_update='dice'):
        losses, kls, pis, reward_losses, reward_losses_before = [], [], [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        self.dice_wts = []

        updated_params=OrderedDict()
        updated_params['z'] = self.z
        for (train_episodes, valid_episodes, _), old_pi in zip(episodes, old_pis):
            reward_loss_before =[]
            for t in range(self.num_updates):
                curr_params, updated_params, reward_loss_before_i = self.adapt(train_episodes[t], updated_params)
                # reward_loss_before.append(reward_loss_before_i)
            with torch.no_grad():
                states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
                rewards_pred = self.reward_net(states,actions,curr_params['z']).squeeze()
                reward_loss = (rewards - rewards_pred)**2
                reward_loss_before.append(reward_loss.mean())
            reward_loss_before = torch.stack(reward_loss_before).reshape((1,-1))
            with torch.set_grad_enabled(old_pi is None):

                #### Policy Objective
                pi, values = self.policy(valid_episodes.observations,self.z_opt[valid_episodes._task_id])#.detach())
                # pi, values = self.policy(valid_episodes.observations,updated_params['z'])#.detach())
                pis.append(detach_distribution(pi))
                dist_loss = pi.entropy().mean()

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                advantages, returns = valid_episodes.gae(values.detach(),tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clamp_param, 1 + self.clamp_param)*advantages
                action_loss = -torch.min(surr1, surr2)
                actions_loss = weighted_mean(action_loss, dim=0,
                    weights=valid_episodes.mask)
                # ipdb.set_trace()
                value_loss = 0.5 * (returns - values.squeeze(2)).pow(2).mean()
                loss = action_loss + self.value_coeff*value_loss + self.entropy_coeff*dist_loss
                losses.append(loss)
                

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

                #### Reward Objective

                states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
                # rewards_pred = self.reward_net(states,actions,self.z_opt[valid_episodes._task_id]).squeeze()
                rewards_pred = self.reward_net(states,actions,updated_params['z']).squeeze()
                reward_loss = (rewards - rewards_pred)**2
                reward_losses.append(reward_loss.mean())
                reward_losses_before.append(reward_loss_before)
        # ipdb.set_trace()
        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), 
                torch.mean(torch.stack(reward_losses,dim=0)),
                torch.mean(torch.stack(reward_losses_before, dim=0), dim=0), pis)

    def test_func(self, episodes):
        losses, kls, pis, reward_losses, reward_losses_before = [], [], [], [], []
        for (train_episodes, valid_episodes) in episodes:
            # reward_loss_before=torch.Tensor(0)
            curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
            states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
            # ipdb.set_trace()
            # updated_params = OrderedDict()
            # updated_params['z'] = torch.ones_like(self.z)
            # updated_params['z'][:self.embed_size//2]*=float(train_episodes.task[0])
            # updated_params['z'][self.embed_size//2:]*=float(train_episodes.task[1])
            rewards_pred = self.reward_net(states,actions,updated_params['z']).squeeze()
            reward_loss = (rewards - rewards_pred)**2
            reward_losses.append(reward_loss.mean())
            reward_losses_before.append(reward_loss_before)
        return  (torch.mean(torch.stack(reward_losses,dim=0)),
                torch.mean(torch.stack(reward_losses_before,dim=0)))

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        self.check = True
        old_loss, _, reward_loss_after, reward_loss_before, old_pis = self.surrogate_loss_rewardnet(episodes, exp_update='dice')
        # old_loss=0
        # reward_loss_after, reward_loss_before= self.test_func(episodes)

        # pdb.set_trace()
        # self.conjugate_gradient_update(episodes, max_kl, cg_iters, cg_damping,            #### TODO: Deprecated, won't work. Fix the TODO below first.
        #                                         ls_max_steps, ls_backtrack_ratio)
        grad_vals = self.gradient_descent_update(old_loss,reward_loss_after*1, episodes)
        # ipdb.set_trace()
        return ((reward_loss_before, reward_loss_after)), old_loss, grad_vals

    def gradient_descent_update(self, old_loss, reward_loss, episodes):
        self.z_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.exp_optimizer.zero_grad()
        self.iter+=1
        # if self.iter%2==1:
        #     ipdb.set_trace()
        
        # (old_loss).backward(retain_graph=True)

        # dice_grad = torch.autograd.grad(old_loss,self.dice_wts,retain_graph=True)
        # # ipdb.set_trace()
        # dice_grad_baselines = []
        # for i, (train_episode, _) in enumerate(episodes):
        #     self.exp_baseline.fit(train_episode, dice_grad[i])
        #     dice_grad_baselines.append(self.exp_baseline(train_episode))
        # # ipdb.set_trace()
        # dice_wts_grad = [grad.detach()-base.squeeze(2).detach() for grad,base in zip(dice_grad,dice_grad_baselines)]
        # dice_grad_sum = 0
        # for i in range(len(self.dice_wts)):
        #     dice_grad_sum+=dice_wts_grad[i]*self.dice_wts[i]
        # dice_grad_sum.sum().backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.exp_policy.parameters(),self.clip)
        # self.exp_optimizer.step()

        wts = 1#math.exp(-self.iter/5)
        (wts*reward_loss+old_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),self.clip)
        nn.utils.clip_grad_norm_(self.reward_net.parameters(),self.clip)
        nn.utils.clip_grad_norm_([self.z_old],self.clip)
        nn.utils.clip_grad_norm_(self.exp_policy.parameters(),self.clip)
        grad_vals = [self.z_old.grad.abs().mean().item()
                    , self.policy.layer_pre1.weight.grad.abs().mean().item()
                    ,self.exp_policy.layer_pre1.weight.grad.abs().mean().item()
                    ,self.reward_net.layer_pre1.weight.grad.abs().mean().item()]
        print("z_grad", "{:.9f}".format(grad_vals[0]))
        print("policy_grad", "{:.9f}".format(grad_vals[1]))
        print("exp_policy_grad", "{:.9f}".format(grad_vals[2]))
        print("reward_grad", "{:.9f}".format(grad_vals[3]))
        self.z_optimizer.step()
        self.reward_optimizer.step()
        self.policy_optimizer.step()
        self.exp_optimizer.step()
        return grad_vals

    def conjugate_gradient_update(self, grads, episodes, max_kl, cg_iters, 
                                  cg_damping, ls_max_steps, ls_backtrack_ratio):

        policy_params = [param for param in self.policy.parameters()]
        exp_params = [param for param in self.exp_policy.parameters()]
        reward_params = [param for param in self.reward_net.parameters()] 
        params = policy_params + exp_params + reward_params + [self.z]
        full_grads = torch.autograd.grad(old_loss, params)
        policy_exp_grads = full_grads[:len(policy_params)+len(exp_params)]
        reward_z_grads = full_grads[len(policy_params)+len(exp_params):]
        for param, grad in zip(policy_params+exp_params, policy_exp_grads):
            param.grad = grad
        self.policy_optimizer.step()
        self.exp_optimizer.step()

        reward_z_grads = full_grads[len(policy_params)+len(exp_params):]        #### TODO: Need to fix this for this specific case. Until then work with Adam.
        grads = parameters_to_vector(rewad_z_grads)
        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(reward_params+[self.z])

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,        #### TODO: Convert this to ProMP optimizer for convenience and better convergence.
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis, exp_update='None')
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, reward_params+[self.z])
        # pdb.set_trace()
        # self.update_exp_policy_dice(old_loss, full_grads[7:])        #### TODO : This should maybe go before the line search to make that update more stable
        #                                                              ####        by adding accounting for the exp_policy update in the inner_update step

    def update_exp_policy_dice(self,old_loss, grads):
        self.exp_optimizer.zero_grad()
        for param, gradient in zip(self.exp_policy.parameters(),grads):
            param.grad = gradient
        # pdb.set_trace()
        self.exp_optimizer.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.exp_policy.to(device, **kwargs)
        self.exp_baseline.to(device, **kwargs)
        self.reward_net.to(device, **kwargs)
        self.z = self.z_old.to(device, **kwargs)
        self.device = device
