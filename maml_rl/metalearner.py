import pdb
import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim

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
    def __init__(self, sampler, policy, exp_policy, baseline, exp_baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.exp_policy = exp_policy
        self.baseline = baseline
        self.exp_baseline = exp_baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)
        self.lr = 7e-4
        self.eps = 1e-5
        self.exp_optimizer = optim.Adam(self.exp_policy.parameters(), lr=self.lr, eps=self.eps)

    def inner_loss(self, episodes, exp_update='dice', params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        exp_pi = self.exp_policy(episodes.observations)
        exp_log_probs_non_diff = episodes.action_probs
        exp_log_probs_diff = torch.zeros_like(exp_log_probs_non_diff)
        log_probs = pi.log_prob(episodes.actions)       #### TODO: Think of better objectives (predicting the reward function, hypothesis testing etc. which are denser)
                                                        ####       Sparse rewards don't make sense for GD especially likelihood based GD. 
                                                        ####       Why even max^m the llhood wrt to actions taken by exploration agent.
                                                        ####       also log_probs of unlikely actions would explode (esp given the off policy setting)
                                                        ####       Hence need clipping etc. at least as in PPO etc.
        if exp_update=='dice':
            exp_log_probs_diff = exp_pi.log_prob(episodes.actions)
            dice_wts = torch.exp(exp_log_probs_diff - exp_log_probs_non_diff)  #### TODO: This might be high variance so reconsider it later maybe.
            log_probs *= dice_wts
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
            exp_log_probs_diff = torch.sum(exp_log_probs_diff, dim=2)
            exp_log_probs_non_diff = torch.sum(exp_log_probs_non_diff, dim=2)
        wts = episodes.mask * torch.exp(log_probs.detach()-exp_log_probs_non_diff)     #### TODO: Needs more thorough investigation : May require better importance wts or find better alternatives
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=wts)

        return loss

    def adapt(self, episodes, first_order=False, exp_update='dice'):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, exp_update)
        # Get the new parameters after a one-step gradient update
        curr_params, updated_params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return curr_params, updated_params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            # ipdb.set_trace()
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.exp_policy, task,
                gamma=self.gamma, device=self.device)

            curr_params, updated_params = self.adapt(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy, task, params=updated_params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            curr_params, updated_params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=updated_params)

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
            curr_params, updated_params = self.adapt(train_episodes)
            # old_pi = curr_params
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=updated_params)
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

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes, exp_update='dice')
        # pdb.set_trace()
        params = [param for param in self.policy.parameters()] + [param for param in self.exp_policy.parameters()]
        full_grads = torch.autograd.grad(old_loss, params)
        grads = full_grads[:7]
        grads = parameters_to_vector(grads)

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
        old_params = parameters_to_vector(self.policy.parameters())

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
            vector_to_parameters(old_params, self.policy.parameters())
        # pdb.set_trace()
        self.update_exp_policy_dice(old_loss, full_grads[7:])        #### TODO : This should maybe go before the line search to make that update more stable
                                                ####        by adding accounting for the exp_policy update in the inner_update step

    def update_exp_policy_dice(self,old_loss, grads):
        self.exp_optimizer.zero_grad()
        for param, gradient in zip(self.exp_policy.parameters(),grads):
            param.grad = gradient
        # pdb.set_trace()
        self.exp_optimizer.step()
        

    # def update_exp_policy_ga(episodes):
        

    #     pass

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.exp_policy.to(device, **kwargs)
        self.exp_baseline.to(device, **kwargs)
        self.device = device
