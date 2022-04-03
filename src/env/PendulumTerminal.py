import numpy as np


class PendulumTerminal:
    def __init__(self, initial_state=np.array([0, np.pi, 0]), dt=0.2, terminal_time=5, inner_step_n=2,
                 action_min=np.array([-2]), action_max=np.array([2])):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

        self.g = 9.8
        self.m = 1.
        self.l = 1.

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            self.state = self.state + np.array([1, self.state[2],
                                                - 3 * self.g / (2 * self.l) * np.sin(self.state[1] + np.pi)
                                                + 3. / (self.m * self.l ** 2) * action[0]]) * self.inner_dt

        if self.state[0] >= self.terminal_time:
            reward = - (np.cos(self.state[1]) - 1) ** 2 - 0.1 * self.state[2] ** 2
            # reward = - np.abs(self.state[1]) - 0.1 * np.abs(self.state[2])
            # print(self.state[1])
            done = True
        else:
            # reward = - 0.01 * (action[0] ** 2) * self.dt
            reward = 0
            done = False

        return self.state, reward, done, None

    def f(self, t, x, u):
        return np.array([x[1],
                         - 3 * self.g / (2 * self.l) * np.sin(x[0] + np.pi)
                         + 3. / (self.m * self.l ** 2) * u[0]])

    def h(self, x):
        return - (np.cos(x[0]) - 1) ** 2 - 0.1 * x[1] ** 2