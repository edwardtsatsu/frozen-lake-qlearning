import os

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states=64,
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9997,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.Q[state]
        max_q = np.max(q)
        candidates = np.where(q == max_q)[0]
        return int(np.random.choice(candidates))

    def update(self, state, action, reward, next_state, done):
        current = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
