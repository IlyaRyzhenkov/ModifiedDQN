import src.DQNAgent
import src.UpdatedDQNAgent
from src.env import simpleControlProblemDiscrete


def run_dqn(env, agent, episode_n=100):
    episode_stat = []
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        for t in range(500):
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
    dt = 0.05
    simpleEnv = simpleControlProblemDiscrete.SimpleControlProblemDiscrete(dt=dt)

    agent = src.DQNAgent.DQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim, session_duration=simpleEnv.steps)
    run_dqn(simpleEnv, agent)
    print("2 algo")

    upd_agent = src.UpdatedDQNAgent.UpdatedDQNAgent(action_n=simpleEnv.action_n, state_dim=simpleEnv.state_dim, session_duration=simpleEnv.steps, dt=dt)
    run_dqn(simpleEnv, upd_agent)
