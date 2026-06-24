import os

import matplotlib.pyplot as plt
import numpy as np
from agent import QLearningAgent
from environment import MAP, FrozenLakeEnv

RESULTS_DIR = "results"
N_EPISODES = 20000
WINDOW = 500
MAX_STEPS = 200  # per-episode step cap to prevent infinite loops

_ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def run_training(alpha, gamma, epsilon_decay=0.9997, n_episodes=N_EPISODES):
    env = FrozenLakeEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)

    rewards = []
    successes = []
    epsilons = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards.append(total_reward)
        successes.append(1 if total_reward > 0 else 0)
        epsilons.append(agent.epsilon)

    return agent, rewards, successes, epsilons


def _rolling(data, window):
    return [np.mean(data[max(0, i - window) : i + 1]) for i in range(len(data))]


def plot_training(rewards, successes, epsilons, window=WINDOW):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(_rolling(rewards, window), color="steelblue")
    axes[0].set_title(f"Reward (rolling avg {window} eps)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Avg Reward")

    axes[1].plot([x * 100 for x in _rolling(successes, window)], color="seagreen")
    axes[1].set_title(f"Success Rate (rolling {window} eps)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 100)

    axes[2].plot(epsilons, color="coral")
    axes[2].set_title("Epsilon Decay")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "training_results.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Graph saved to {out}")


def extract_policy(Q):
    policy = []
    for state in range(64):
        row, col = divmod(state, 8)
        cell = MAP[row][col]
        if cell == "H":
            policy.append("H")
        elif cell == "G":
            policy.append("G")
        else:
            # 'S' and 'F' cells: show best action
            policy.append(_ACTION_SYMBOLS[int(np.argmax(Q[state]))])
    return policy


def print_policy(policy):
    print("\nLearned Policy:")
    print("+" + "----" * 8 + "+")
    for row in range(8):
        line = "| "
        for col in range(8):
            line += policy[row * 8 + col] + "  "
        print(line + "|")
    print("+" + "----" * 8 + "+")


def _summarise(label, alpha, gamma, successes, n_episodes):
    rate = sum(successes) / n_episodes * 100
    print(
        f"  [{label}] α={alpha}, γ={gamma}  →  success rate: {rate:.1f}%  "
        f"({sum(successes)}/{n_episodes} episodes)"
    )


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("Hyperparameter Experiments")
    print("=" * 55)

    _, _, s1, _ = run_training(alpha=0.3, gamma=0.95, epsilon_decay=0.9995)
    _summarise("fast-learn", 0.3, 0.95, s1, N_EPISODES)

    _, _, s2, _ = run_training(alpha=0.05, gamma=0.99, epsilon_decay=0.9997)
    _summarise("slow-learn", 0.05, 0.99, s2, N_EPISODES)

    print()
    print("=" * 55)
    print("Default Training  (α=0.1, γ=0.99, ε_decay=0.9997)")
    print("=" * 55)
    agent, rewards, successes, epsilons = run_training(
        alpha=0.1, gamma=0.99, epsilon_decay=0.9997
    )
    _summarise("default", 0.1, 0.99, successes, N_EPISODES)

    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    agent.save(q_path)
    print(f"Q-table saved to {q_path}")

    plot_training(rewards, successes, epsilons)

    policy = extract_policy(agent.Q)
    print_policy(policy)
