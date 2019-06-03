import numpy as np
import torch
import torch.nn.functional as F
import ipdb

class BatchEpisodes(object):
    def __init__(self, batch_size, task, corners, gamma=0.95, device='cpu', max_episode_length=200):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._observations_next_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._action_probs_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._mask_list = []
        self.dones = {}

        self._observations = None
        self._observations_next = None
        self._actions = None
        self._action_probs = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self._task = task
        if corners is not None:
            self.corners = corners
        else:
            self.corners = [np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])]
        self._task_id = np.argmax(task==self.corners)
        self.num_samples = 0
        self.max_episode_length = max_episode_length

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                try:
                    observations[:length, i] = np.stack(self._observations_list[i], axis=0)
                except ValueError:
                    ipdb.set_trace()
                    continue
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def observations_next(self):
        if self._observations_next is None:
            observation_shape = self._observations_next_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_next_list[i])
                try:
                    observations[:length, i] = np.stack(self._observations_next_list[i], axis=0)
                except ValueError:
                    ipdb.set_trace()
                    continue
            self._observations_next = torch.from_numpy(observations).to(self.device)
        return self._observations_next

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                try:
                    actions[:length, i] = np.stack(self._actions_list[i], axis=0)
                except ValueError:
                    ipdb.set_trace()
                    continue
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property    
    def action_probs(self):
        if self._action_probs is None:
            action_probs_shape = self._action_probs_list[0][0].shape
            action_probs = np.zeros((len(self), self.batch_size)
                + action_probs_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._action_probs_list[i])
                try:
                    action_probs[:length, i] = np.stack(self._action_probs_list[i], axis=0)
                except ValueError:
                    ipdb.set_trace()
                    continue
            self._action_probs = torch.from_numpy(action_probs).to(self.device)
        return self._action_probs

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                try:
                    rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
                except ValueError:
                    ipdb.set_trace()
                    continue
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    def nstep_returns(self, n):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            rewards_n_shifted = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards_n_shifted[:-n] = rewards[n:].copy()
            mask = self.mask.cpu().numpy()
            mask_n_shifted = np.zeros((len(self), self.batch_size), dtype=np.float32)
            mask_n_shifted[:-n] = mask[n:].copy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i] - rewards_n_shifted[i] * mask_n_shifted[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    @property
    def task(self):
        return self._task

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations, actions, rewards, batch_ids, action_probs, observations_next, dones):
        for observation, action, reward, batch_id, action_prob, observation_next, done in zip(
                observations, actions, rewards, batch_ids, action_probs, observations_next, dones):
            if batch_id is None:
                continue
            while batch_id >= len(self._observations_list):
                self._observations_list.append([])
                self._actions_list.append([])
                self._rewards_list.append([])
                self._action_probs_list.append([])
                self._observations_next_list.append([])

            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))
            self._action_probs_list[batch_id].append(action_prob.astype(np.float32))
            self._observations_next_list[batch_id].append(observation_next.astype(np.float32))
            if done:
                self.num_samples += len(self._rewards_list[batch_id])
                self.prev_batch_id = batch_id
                self.dones[batch_id]=True

    def adjust_to_num_samples(self, prev_batch_ids_to_remove):
        diff = self.num_samples - self.max_episode_length
        self._observations_list[self.prev_batch_id][:-diff]
        self._actions_list[self.prev_batch_id][:-diff]
        self._rewards_list[self.prev_batch_id][:-diff]
        self._action_probs_list[self.prev_batch_id][:-diff]
        self._observations_next_list[self.prev_batch_id][:-diff]
        for batch_id in sorted(prev_batch_ids_to_remove, reverse=True):
            if batch_id!=self.prev_batch_id:
                self._observations_list.pop(batch_id)
                self._actions_list.pop(batch_id)
                self._rewards_list.pop(batch_id)
                self._action_probs_list.pop(batch_id)
                self._observations_next_list.pop(batch_id)
        self.batch_size=len(self._rewards_list)

    def __len__(self):
        return max(map(len, self._rewards_list))