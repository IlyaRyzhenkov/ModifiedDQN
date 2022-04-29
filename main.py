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


def compare_algorithms():
    dt = 0.025
    episode_n = 500
    batch_size = 128
    simpleEnv = simpleControlProblemDiscrete.SimpleControlProblemDiscrete(dt=dt)

    agent = src.DQNAgent.DQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim,
                                  session_duration=simpleEnv.steps, batch_size=batch_size)
    stat_1 = run_dqn(simpleEnv, agent, episode_n=episode_n)
    print("2 algo")

    upd_agent = src.UpdatedDQNAgent.UpdatedDQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim,
                                                    session_duration=simpleEnv.steps, dt=dt, batch_size=batch_size)
    stat_2 = run_dqn(simpleEnv, upd_agent, episode_n=episode_n)

    plt.plot(stat_1, label="DQN")
    plt.plot(stat_2, label="Modified DQN")
    plt.show()


def compare_by_batch_size():
    dt = 0.025
    start_batch = 256
    episode_n = 250
    env = simpleControlProblemDiscrete.SimpleControlProblemDiscrete(dt=dt)
    batch_size = start_batch
    for i in range(4):
        print(f"batch size = {batch_size}")
        agent = src.UpdatedDQNAgent.UpdatedDQNAgent(action_n=env.action_n, state_dim=env.state_dim,
                                                    session_duration=episode_n, dt=dt, batch_size=batch_size)
        stat = run_dqn(env, agent, episode_n=episode_n)
        plt.plot(stat, label=f"batch size = {batch_size}")
        batch_size *= 2
    plt.title(f"Зависимость по batch size. dt={dt}, episodes = {episode_n}")
    plt.legend()
    plt.show()


def compare_by_episode_n(batch_size):
    dt = 0.025
    start_episode_n = 200
    steps = 6
    env = simpleControlProblemDiscrete.SimpleControlProblemDiscrete(dt=dt)
    episode_n = start_episode_n
    for i in range(steps):
        print(f"{episode_n} episodes")
        agent = src.UpdatedDQNAgent.UpdatedDQNAgent(action_n=env.action_n, state_dim=env.state_dim,
                                                    session_duration=episode_n, dt=dt, batch_size=batch_size)
        stat = run_dqn(env, agent, episode_n=episode_n)
        plt.plot(stat, label=f"episodes = {episode_n}")
        episode_n += 100
    plt.title(f"Зависимость от длительности обучения. dt={dt}, batch size = {batch_size}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    compare_by_episode_n(512)
    # compare_by_batch_size()
