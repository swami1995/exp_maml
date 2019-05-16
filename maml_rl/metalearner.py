import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence
import numpy as np
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize, moving_weighted_normalize)
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
    def __init__(self, sampler, policy, exp_policy, baseline, exp_baseline, reward_net, reward_net_outer, exp_baseline_targ, z_old, baseline_type,
                 embed_size=100, gamma=0.95, fast_lr=0.5, tau=1.0, lr=7e-4, eps=1e-5, device='cpu', algo='a2c', use_emaml=False, num_updates_outer=1, use_target=False):
        self.sampler = sampler
        self.policy = policy
        self.exp_policy = exp_policy
        self.baseline = baseline
        self.exp_baseline = exp_baseline
        self.reward_net = reward_net
        self.reward_net_outer = reward_net_outer
        self.exp_baseline_targ = exp_baseline_targ
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.lamda = 0.1

        self.use_target = use_target ## True
        self.baseline_type = baseline_type ## 'lin'
        self.num_updates_outer = num_updates_outer ## 5
        self.reward_type = 'dice_reward' ## or 'env_reward'
        self.moving_normalize = True

        self.lr_r = lr
        self.eps_r = eps
        self.lr_z = lr*0.005
        self.eps_z = eps
        self.lr_p = lr
        self.eps_p = eps
        self.lr_e = lr*0.1
        self.eps_e = eps
        self.lr_ro = lr
        self.eps_ro = eps
        self.lr_eb = lr
        
        self.embed_size = embed_size
        self.alpha = 5e-8
        self.clip = 0.5
        self.clip_param = 0.2
        if z_old is None:
            self.z_old = nn.Parameter(torch.zeros((1,embed_size)))
            nn.init.xavier_uniform_(self.z_old)
        else:
            self.z_old = nn.Parameter(z_old)

        assert algo in ['a2c', 'ppo']
        self.algo = algo
        self.use_emaml = use_emaml

        self.to(device)
        self.z_optimizer = optim.Adam([self.z_old], lr=self.lr_z, eps=self.eps_z)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr_r, eps=self.eps_r)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_p, eps=self.eps_p)
        self.exp_optimizer = optim.Adam(self.exp_policy.parameters(), lr=self.lr_e, eps=self.eps_e)
        self.reward_outer_optimizer = optim.Adam(self.reward_net_outer.parameters(), lr=self.lr_ro, eps=self.eps_ro)
        if self.baseline_type == 'nn':
            self.exp_baseline_optimizer = optim.Adam(self.exp_baseline.parameters(), lr=self.lr_eb, eps=self.eps_e)

        patience = 30000
        min_lr_factor = 0.1
        factor = 0.95
        self.exp_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.exp_optimizer, 'min', patience=patience, factor=factor,
                                                                            min_lr=self.lr_e*min_lr_factor)
        self.z_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.z_optimizer, 'min', patience=patience, factor=factor, 
                                                                          min_lr=self.lr_z*min_lr_factor)
        self.policy_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.policy_optimizer, 'min', patience=patience, factor=factor, 
                                                                               min_lr=self.lr_p*min_lr_factor)
        self.reward_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.reward_optimizer, 'min', patience=patience, factor=factor, 
                                                                               min_lr=self.lr_r*min_lr_factor)
        self.iter = 0
        self.dice_wts = []
        self.dice_wts_detached = []
        self.exp_entropy = []
        self.z_grad_ph = []
        self.updated_params = []
        self.inner_losses = []
        self.exp_pis = []
        self.moving_params_normalize = np.array([0.,1.])
        self.z_exp = torch.zeros_like(self.z)
        self.z_opt = (torch.ones((4,1,embed_size))*self.z_old.clone().detach().unsqueeze(0)).to(device)
        for i in range(4):
            self.z_opt[i,0,i*embed_size//4:(i+1)*embed_size//4] = 1

    def inner_loss(self, episodes, exp_update='dice', params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        states, actions, rewards = episodes.observations, episodes.actions, episodes.returns
        rewards_pred, self.z_ph = self.reward_net(states,actions,self.z, ph=True)
        rewards_pred = rewards_pred.squeeze(-1)
        loss = (rewards - rewards_pred)**2
        loss_diff = torch.abs((rewards - rewards_pred)*2).detach()
        # topkthvalue = torch.kthvalue(loss.cpu(), 9*rewards_pred.shape[0]//10, dim=0, keepdim=True)
        # loss = torch.ge(loss, topkthvalue.to(self.device)).float()*loss
        exp_pi = self.exp_policy(states, self.z_exp.detach())#self.z.detach()) 
        self.exp_pis.append(exp_pi)
        exp_log_probs_non_diff = episodes.action_probs
        exp_log_probs_diff = torch.zeros_like(exp_log_probs_non_diff)
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
            dice_wts = torch.exp(exp_log_probs_diff.sum(dim=2) - exp_log_probs_non_diff.sum(dim=2))
            self.dice_wts.append(dice_wts)
            self.dice_wts_detached.append(dice_wts.detach())
            self.dice_wts_detached[-1].requires_grad_()
            # cum_wts = torch.exp(torch.log(self.dice_wts_detached[-1]).cumsum(dim=0))
            self.inner_losses.append(loss_diff.detach())
            loss *= self.dice_wts_detached[-1]
            # loss *= cum_wts

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
        grads = torch.autograd.grad(loss, self.z_ph, create_graph=not first_order)
        self.z_grad_ph.append(grads)
        updated_params = OrderedDict()
        curr_params = OrderedDict()
        updated_params['z'] = self.z - step_size * grads[0].sum(0).sum(0)
        self.updated_params.append(updated_params['z'])
        curr_params['z'] = self.z
        return curr_params, updated_params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            curr_params = OrderedDict()
            curr_params['z'] = self.z_exp
            train_episodes = self.sampler.sample(self.exp_policy, task, params=curr_params, gamma=self.gamma, device=self.device)

            curr_params, updated_params, _ = self.adapt(train_episodes, first_order=first_order)
            # updated_params['z'] = self.z_opt[train_episodes._task_id]
            valid_episodes = self.sampler.sample(self.policy, task, params=updated_params, gamma=self.gamma, device=self.device)
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

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
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
                pi = self.policy(valid_episodes.observations,updated_params['z'])#, params=updated_params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), pis)

    def surrogate_loss_rewardnet(self, episodes, old_pis=None, exp_update='dice'):
        losses, kls, pis, reward_losses, reward_losses_before, reward_losses_inner = [], [], [], [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        self.dice_wts = []
        self.dice_wts_detached = []
        self.exp_entropy = []
        self.z_grad_ph = []
        self.updated_params = []
        self.inner_losses = []
        self.exp_pis = []
        self.baseline_exp = 0
        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
            reward_losses_inner.append(reward_loss_before)
            self.baseline.fit(valid_episodes)
            # old_pi = curr_params
            with torch.no_grad():
                states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.returns
                rewards_pred = self.reward_net_outer(states,actions,curr_params['z']).squeeze(-1)
                reward_loss = (rewards - rewards_pred)**2
                reward_loss_before = reward_loss.mean()
            with torch.set_grad_enabled(old_pi is None):

                # Policy Objective
                pi = self.policy(valid_episodes.observations, updated_params['z'])
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)
                
                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions) - valid_episodes.action_probs)
                      
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                self.baseline_exp = valid_episodes.rewards.sum(0)
                # self.baseline_exp+=torch.log(valid_episodes.rewards.sum(0)+1)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advantages
                action_loss = torch.min(surr1, surr2)
                loss = -weighted_mean(action_loss, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

                # Reward Objective
                states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.returns
                rewards_pred = self.reward_net_outer(states,actions,updated_params['z'].detach()).squeeze(-1)
                reward_loss = (rewards - rewards_pred)**2
                reward_losses.append(reward_loss.mean())
                reward_losses_before.append(reward_loss_before)
        
        self.baseline_exp/=(len(episodes)*valid_episodes.rewards.shape[1])
        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), 
                torch.mean(torch.stack(reward_losses,dim=0)),
                torch.mean(torch.stack(reward_losses_before,dim=0)),
                torch.mean(torch.stack(reward_losses_inner,dim=0)), pis)

    def test_func(self, episodes):
        reward_losses, reward_losses_before = [], []
        for (train_episodes, valid_episodes) in episodes:
            # reward_loss_before=torch.Tensor(0)
            curr_params, updated_params, reward_loss_before = self.adapt(train_episodes)
            states, actions, rewards = valid_episodes.observations, valid_episodes.actions, valid_episodes.rewards
            # updated_params = OrderedDict()
            # updated_params['z'] = torch.ones_like(self.z)
            # updated_params['z'][:self.embed_size//2]*=float(train_episodes.task[0])
            # updated_params['z'][self.embed_size//2:]*=float(train_episodes.task[1])
            rewards_pred = self.reward_net(states,actions,updated_params['z']).squeeze()
            reward_loss = (rewards - rewards_pred)**2
            reward_losses.append(reward_loss.mean())
            reward_losses_before.append(reward_loss_before)
        return (torch.mean(torch.stack(reward_losses,dim=0)), torch.mean(torch.stack(reward_losses_before,dim=0)))

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        self.old_exp_pis = []
        for j in range(self.num_updates_outer):
            old_loss, _, reward_loss_after, reward_loss_before, reward_loss_inner, old_pis = self.surrogate_loss_rewardnet(episodes, exp_update='dice')
            # old_loss=0
            # reward_loss_after, reward_loss_before= self.test_func(episodes)
            
            # TODO: Depcretaed, won't work. Fix the TODO below first.
            # self.conjugate_gradient_update(episodes, max_kl, cg_iters, cg_damping,            
            #                                         ls_max_steps, ls_backtrack_ratio)
            grad_vals = self.gradient_descent_update(old_loss*10,reward_loss_after*1, reward_loss_inner, episodes, j)
        return ((reward_loss_before, reward_loss_after)), old_loss, grad_vals

    def gradient_descent_update(self, old_loss, reward_loss, reward_loss_inner, episodes, j):
        self.z_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        self.reward_outer_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.exp_optimizer.zero_grad()
        self.exp_optimizer_scheduler.step(old_loss)
        self.reward_optimizer_scheduler.step(old_loss)
        self.policy_optimizer_scheduler.step(old_loss)
        self.z_optimizer_scheduler.step(old_loss)
        self.iter+=1
        wts = math.exp(-self.iter/5)
        wts1 = 1
        dice_wts_grad = []

        dice_grad = torch.autograd.grad(old_loss, self.updated_params, retain_graph=True)
        # dice_grad = torch.autograd.grad(old_loss+wts*reward_loss,self.updated_params,retain_graph=True)   
        # dice_grad_detached = torch.autograd.grad(old_loss,self.dice_wts_detached,retain_graph=True)
        
        rewards_exp = []
        kls = []
        kl_grads = 0.
        value_loss = []
        # if j>4:
        if self.use_emaml:
            for i, (train_episodes, valid_episodes) in enumerate(episodes):
                # TODO: check if log scaling is useful?
                # returns = torch.log(valid_episodes.rewards.sum(0, keepdim=True)+1)
                
                returns = valid_episodes.rewards.sum(0, keepdim=True)
                advantages = returns - self.baseline_exp
                ratio = self.dice_wts[i]
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                # ppo loss
                if self.algo == 'ppo':
                    surr1 = ratio * advantages.detach()
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
                    action_loss = -torch.min(surr1, surr2)
                
                # a2c loss
                if self.algo == 'a2c':
                    action_loss = -ratio*advantages.detach()
                
                dice_wts_grad.append(action_loss)
            dice_grad_mean = 0
            for i in range(len(self.dice_wts)):
                dice_grad_mean+=dice_wts_grad[i].mean(0).sum()/(action_loss.shape[1]*len(self.dice_wts))
            dice_grad_mean.backward()

        else:
            for i, (train_episodes, valid_episodes) in enumerate(episodes):
                # normalized_entropy = weighted_normalize(self.exp_entropy[i])
                # dice_grad_normalized = weighted_normalize(dice_grad[i]).detach()
                # returns = - dice_grad_normalized #+ weighted_normalize(train_episodes.rewards) #+ normalized_entropy 
                if self.reward_type == 'dice_reward':
                    rewards_exp.append(self.fast_lr*torch.sum(dice_grad[i].unsqueeze(0)*self.z_grad_ph[i][0], dim=-1)/self.inner_losses[i]) 
                ### NOTE : The reward depends on other networks (ENV?) hence it changes with time, what modifications to the standard value computations should we make?
                ### NOTE : Maybe use a target network?
                elif self.reward_type == 'env_reward':
                    rewards_exp.append(train_episodes.rewards)
                # reward_detatched = dice_grad_detached[i]
                if self.moving_normalize:
                    normalized_rewards, self.moving_params_normalize = moving_weighted_normalize(rewards_exp[-1], no_mean=True, moving_params=self.moving_params_normalize) 
                else:
                    normalized_rewards, self.moving_params_normalize = moving_weighted_normalize(rewards_exp[-1], no_mean=True)#, moving_params=self.moving_params_normalize) 
                ### NOTE : Does it make sense to write normalized rewards for a value function? Maybe keep a moving average or something like bnorm? 
                ### But bnorm isn't stable in RL --> investigate!
                
                if self.baseline_type=='lin':
                    returns = self.get_returns(normalized_rewards, train_episodes.mask).detach()
                    self.exp_baseline.fit(train_episodes, returns)
                    values = self.exp_baseline(train_episodes)
                    
                elif self.baseline_type=='nn':
                    values = self.exp_baseline(train_episodes.observations)
                    if self.use_target:
                        values_targ = self.exp_baseline_targ(train_episodes.observations)
                        _, returns_targ = self.gae(values_targ,normalized_rewards, tau=self.tau, mask=train_episodes.mask)

                # advantages = returns - values.squeeze()
                advantages, returns = self.gae(values,normalized_rewards, tau=self.tau, mask=train_episodes.mask)
                # TODO: Perform experiments with normalized advantages as well
                advantages = weighted_normalize(advantages, weights=train_episodes.mask) 
                if j==0:
                    self.old_exp_pis.append(detach_distribution(self.exp_pis[i]))
                if self.algo == 'ppo':
                    ratio = self.dice_wts[i]
                    surr1 = ratio * advantages.detach()
                    surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advantages.detach()
                    action_loss = -torch.min(surr1, surr2)
                elif self.algo == 'trpo':
                    surr1 = ratio * advantages.detach()
                    action_loss = -surr1
                    kl = weighted_mean(kl_divergence(self.exp_pis[i], self.old_exp_pis[i]), dim=0, weights=mask)
                    kls.append(kl)

                if self.baseline_type=='nn':
                    if self.use_target:
                        value_loss.append((values.squeeze(-1) - returns_targ.detach()).pow(2).mean(0) * 0.5)
                    else:
                        value_loss.append((values.squeeze(-1) - returns.detach()).pow(2).mean(0) * 0.5)
                # if j==4:
                #     ipdb.set_trace()
                mask = train_episodes.mask
                if train_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                dice_wts_grad.append(action_loss)

            dice_grad_mean = 0
            for i in range(len(self.dice_wts)):
                dice_grad_mean+=dice_wts_grad[i].sum()#/len(self.dice_wts)
            # scale = torch.sum(torch.tensor([dice_grad[i].sum() for i in range(len(dice_grad))]))/dice_grad_mean.sum().item()
            if dice_grad_mean==0:
                scale=torch.tensor(1)
            else:
                scale = torch.abs(torch.sum(torch.tensor([rewards_exp[i].sum() for i in range(len(rewards_exp))]))/dice_grad_mean.sum().item())
            dice_grad_mean=dice_grad_mean*scale.item()
            if np.isnan(dice_grad_mean.item()):
                self.exp_policy.layer_pre1.weight.grad=torch.zeros_like(self.exp_policy.layer_pre1.weight)
            else:
                dice_grad_mean = dice_grad_mean#+torch.mean(torch.stack(kls, dim=0))*0.1
                if self.algo=='trpo':
                    (torch.mean(torch.stack(kls, dim=0))*0.1).backward(retain_graph=True)
                    kl_grads = self.exp_policy.layer_pre1.weight.grad.abs().mean().item()
                if self.baseline_type=='nn':
                    dice_grad_mean= torch.mean(torch.stack(value_loss, dim=0)) + dice_grad_mean
                dice_grad_mean.backward()
                if np.isnan(self.exp_policy.layer_pre1.weight.grad.abs().mean().item()):
                    self.exp_policy.layer_pre1.weight.grad=torch.zeros_like(self.exp_policy.layer_pre1.weight)
        
        (old_loss+wts1*reward_loss).backward()
        # old_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),self.clip)
        nn.utils.clip_grad_norm_(self.reward_net.parameters(),self.clip)
        nn.utils.clip_grad_norm_([self.z_old],self.clip)
        nn.utils.clip_grad_norm_(self.exp_policy.parameters(),self.clip)
        nn.utils.clip_grad_norm_(self.reward_net_outer.parameters(),self.clip)

        grad_vals = [self.z_old.grad.abs().mean().item(),
                     self.policy.layer_pre1.weight.grad.abs().mean().item(),
                     self.exp_policy.layer_pre1.weight.grad.abs().mean().item(),
                     self.reward_net.layer_pre1.weight.grad.abs().mean().item(),
                     self.reward_net_outer.layer_pre1.weight.grad.abs().mean().item(),
                     kl_grads]
        print("z_grad {:.9f}".format(grad_vals[0]))
        print("policy_grad {:.9f}".format(grad_vals[1]))
        print("exp_policy_grad {:.9f}".format(grad_vals[2]))
        print("kl_grads {:.9f}".format(grad_vals[5]))
        print("reward_grad {:.9f}".format(grad_vals[3]))
        print("reward_grad_outer {:.9f}\n".format(grad_vals[4]))
        self.z_optimizer.step()
        self.reward_optimizer.step()
        self.reward_outer_optimizer.step()
        self.policy_optimizer.step()
        self.exp_optimizer.step()
        if self.baseline_type=='nn':
            self.exp_baseline_optimizer.step()
        # self.exp_optimizer_scheduler.step()
        return grad_vals
    
    def update_targs_value(self):
        # Soft update of critic_target
        for parameter_target, parameter_source in zip(self.exp_baseline_targ.parameters(), self.exp_baseline.parameters()):
            parameter_target.data.copy_((1 - self.lamda) * parameter_target.data + self.lamda * parameter_source.data)

    def update_exp_policy_dice(self,old_loss, grads):
        self.exp_optimizer.zero_grad()
        for param, gradient in zip(self.exp_policy.parameters(),grads):
            param.grad = gradient
        self.exp_optimizer.step()

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.exp_policy.to(device, **kwargs)
        self.exp_baseline.to(device, **kwargs)
        self.reward_net.to(device, **kwargs)
        self.reward_net_outer.to(device, **kwargs)
        self.exp_baseline_targ.to(device, **kwargs)
        self.z = self.z_old.to(device, **kwargs)
        self.device = device

    def get_returns(self, rewards, mask):
        return_ = torch.zeros(rewards.shape[1]).to(self.device)
        returns = torch.zeros(rewards.shape[:2]).to(self.device)
        # rewards = rewards.cpu().numpy()
        for i in range(len(rewards) - 1, -1, -1):
            returns[i] = self.gamma * return_ + rewards[i] * mask[i]
        return returns

    def gae(self, values, rewards, tau=1.0, mask=None):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values, (0, 0, 0, 1))
        if mask is None:
            mask = torch.ones_like(values[1:])

        deltas = rewards + self.gamma * values[1:]*mask - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(rewards) - 1, -1, -1):
            gae = gae * self.gamma * tau * mask[i]+ deltas[i]
            advantages[i] = gae 
        returns = values[:-1] + advantages
        return advantages, returns
