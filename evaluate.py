import numpy as np
from agent import QLearningAgent
from environment import FrozenLakeEnv

Q_TABLE_PATH = "results/q_table.npy"
N_EVAL_EPISODES = 1000
MAX_STEPS = 200


def evaluate(n_episodes=N_EVAL_EPISODES):
    agent = QLearningAgent(epsilon=0.0)
    agent.load(Q_TABLE_PATH)

    env = FrozenLakeEnv()
    total_reward = 0.0
    successes = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            steps += 1

        total_reward += ep_reward
        if ep_reward > 0:
            successes += 1

    failures = n_episodes - successes
    success_rate = successes / n_episodes * 100
    avg_reward = total_reward / n_episodes

    print("=" * 35)
    print("      Evaluation Report")
    print("=" * 35)
    print(f"  Episodes evaluated : {n_episodes}")
    print(f"  Successful runs    : {successes}")
    print(f"  Failures           : {failures}")
    print(f"  Success rate       : {success_rate:.1f}%")
    print(f"  Average reward     : {avg_reward:.4f}")
    print("=" * 35)


if __name__ == "__main__":
    evaluate()
