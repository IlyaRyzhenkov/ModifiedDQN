import src.DQNAgent
import src.UpdatedDQNAgent
from src.env import simpleControlProblemDiscrete
import matplotlib.pyplot as plt


def run_dqn(env, agent, episode_n=150):
    episode_stat = []
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        for t in range(env.steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.fit(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        episode_stat.append(total_reward)
        print(total_reward)
    return episode_stat


if __name__ == '__main__':
    dt = 0.025
    episode_n = 500
    simpleEnv = simpleControlProblemDiscrete.SimpleControlProblemDiscrete(dt=dt)

    agent = src.DQNAgent.DQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim, session_duration=simpleEnv.steps)
    stat_1 = run_dqn(simpleEnv, agent, episode_n=episode_n)
    print("2 algo")

    upd_agent = src.UpdatedDQNAgent.UpdatedDQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim, session_duration=simpleEnv.steps, dt=dt)
    stat_2 = run_dqn(simpleEnv, upd_agent, episode_n=episode_n)

    plt.plot(stat_1, label="DQN")
    plt.plot(stat_2, label="Modified DQN")
    plt.show()

