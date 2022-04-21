from copy import deepcopy

import numpy as np
from torch import nn
import torch
import random
from src import network


class UpdatedDQNAgent(nn.Module):
    def __init__(self, state_dim, action_n, session_duration, dt, batch_size=64, val_tau=1e-2, adv_tau=1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.session_duration = session_duration
        self.dt = dt
        # Коэффициент шума при выборе действия
        self.epsilon_delta = 1 / session_duration
        self.epsilon = 1

        self.gamma = 1
        self.memory_size = 10000
        self.memory = []
        self.batch_size = batch_size
        self.learinig_rate = 1e-2

        self.val = network.Network(self.state_dim, 1)
        self.target_val = deepcopy(self.val)
        self.adv = network.Network(self.state_dim, self.action_n)
        self.target_adv = deepcopy(self.adv)
        self.val_optimizer = torch.optim.Adam(self.val.parameters(), lr=self.learinig_rate)
        self.adv_optimizer = torch.optim.Adam(self.adv.parameters(), lr=self.learinig_rate)

        # Коэффициенты при скользящем улучшении весов
        self.val_tau = val_tau
        self.adv_tau = adv_tau

    def get_action(self, state):
        actions = np.arange(self.action_n)
        max_action = self.calculate_max_action(state)
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[max_action] += 1 - self.epsilon
        for i in range(probs.size):
            if probs[i] < 0:
                probs[i] = 0
        action = np.random.choice(actions, p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        if len(self.memory) > self.batch_size:
            """ Optimizes using the DAU variant of advantage updating.
                Note that this variant uses max_action, and not max_next_action, as is
                more common with standard Q-Learning. It relies on the set of equations
                V^*(s) + dt A^*(s, a) = r(s, a) dt + gamma^dt V^*(s)
                A^*(s, a) = adv_function(s, a) - adv_function(s, max_action)
            """
            batch = random.sample(self.memory, self.batch_size)

            states, actions, rewards, dones, next_states = list(zip(*batch))
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)

            v = self.val(states)
            next_v = self.calculate_next_v(dones, next_states)
            pre_advs_data = self.adv(states)
            pre_advs = torch.FloatTensor(self.batch_size)
            pre_max_advs = torch.FloatTensor(self.batch_size)
            for i in range(self.batch_size):
                max_action = torch.argmax(pre_advs_data[i])
                pre_advs[i] = pre_advs_data[i][actions[i]]
                pre_max_advs[i] = pre_advs_data[i][max_action]
            adv = pre_advs - pre_max_advs
            adv = adv.reshape(self.batch_size, 1)
            q_values = v + self.dt * adv
            expected_q_values = (rewards + (self.gamma ** self.dt) * next_v).detach()
            critic_value = (q_values - expected_q_values) ** 2

            loss = critic_value.mean()
            self.update_target_models(self.target_val, self.val, self.val_optimizer, self.target_adv, self.adv,
                                      self.adv_optimizer, loss)

            if self.epsilon > 0.01:
                self.epsilon -= self.epsilon_delta

    def calculate_max_action(self, state):
        state = torch.FloatTensor(state)
        max_action = torch.argmax(self.adv(state))
        return max_action

    def calculate_next_v(self, dones, next_states):
        next_v = self.target_val(next_states)
        for i in range(self.batch_size):
            if dones[i]:
                next_v[i] = 0
        return next_v

    def update_target_models(self, target_val, val, val_optimizer, target_adv, adv, adv_optimizer, loss):
        val_optimizer.zero_grad()
        adv_optimizer.zero_grad()
        loss.backward()
        val_optimizer.step()
        adv_optimizer.step()
        for val_target_param, val_param in zip(target_val.parameters(), val.parameters()):
            val_target_param.data.copy_((1 - self.val_tau) * val_target_param.data + self.val_tau * val_param)
        for adv_target_param, adv_param in zip(target_adv.parameters(), adv.parameters()):
            adv_target_param.data.copy_((1 - self.adv_tau) * adv_target_param.data + self.adv_tau * adv_param)
        return None
