import numpy as np


class SimpleControlProblemDiscrete:
    def __init__(self, dt=0.05, terminal_time=2, initial_state=np.array([0, 1]),
                 action_values=np.array([[-1], [0], [1]])):
        self.state_dim = 2
        self.action_values = action_values
        self.action_n = self.action_values.size
        self.dt = dt
        self.steps = int(terminal_time / dt) + 1
        self.terminal_time = terminal_time
        self.initial_state = initial_state
        self.state = self.reset()

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        _action = self.action_values[action]
        self.state = self.state + np.array([1, _action[0]]) * self.dt
        reward = - 0.5 * _action[0] ** 2 * self.dt
        done = False
        if self.state[0] >= self.terminal_time:
            reward -= self.state[1] ** 2
            done = True
        return self.state, reward, done, None
