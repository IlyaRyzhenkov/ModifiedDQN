import numpy as np
from torch import nn
import torch
import random
from src import network


class UpdatedDQNAgent(nn.Module):
    def __init__(self, state_dim, action_n, session_duration, dt):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.session_duration = session_duration
        self.dt = dt
        self.epsilon_delta = 1 / session_duration
        self.epsilon = 1

        self.gamma = 0.95
        self.memory_size = 10000
        self.memory = []
        self.batch_size = 64
        self.learinig_rate = 1e-2

        self.val = network.Network(self.state_dim, self.action_n)
        #TODO уточнить про размерности
        self.adv = network.Network(self.state_dim + 1, 1)
        self.val_optimizer = torch.optim.Adam(self.val.parameters(), lr=self.learinig_rate)
        self.adv_optimizer = torch.optim.Adam(self.adv.parameters(), lr=self.learinig_rate)

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
        max_action = self.calculate_max_action(state)
        self.memory.append([state, action, max_action, reward, done, next_state])
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

            states, actions, max_actions, rewards, dones, next_states = list(zip(*batch))
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            max_actions = torch.FloatTensor(max_actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)

            v = self.val(states)
            next_v = self.calculate_next_v(dones, next_states)
            pre_advs = self.adv(torch.cat((states, actions), 1))
            pre_max_advs = self.adv(torch.cat((states, max_actions), 1))
            adv = pre_advs - pre_max_advs

            q_values = v + self.dt * adv
            expected_q_values = (rewards * self.dt + self.gamma ** self.dt * next_v).detach()
            critic_value = (q_values - expected_q_values) ** 2

            self.val_optimizer.zero_grad()
            self.adv_optimizer.zero_grad()
            critic_value.mean().backward()
            self.val_optimizer.step()
            self.adv_optimizer.step()

            if self.epsilon > 0.01:
                self.epsilon -= self.epsilon_delta

    def calculate_max_action(self, state):
        state = torch.FloatTensor(state)
        actions = np.arange(self.action_n)
        max_action = 0
        max_adv = -1000000000000
        for action in actions:
            adv = self.adv(torch.cat((state, torch.FloatTensor([action]))))
            if adv > max_adv:
                max_adv = adv
                max_action = action
        return max_action

    def calculate_next_v(self, dones, next_states):
        next_v = self.val(next_states)
        for i in range(self.batch_size):
            if dones[i]:
                next_v[i] = 0
        return next_v


    def calculate_expected_q(self, q_values, rewards, next_v):
        targets = q_values.clone()
        for i in range(self.batch_size):
            pass
        pass

