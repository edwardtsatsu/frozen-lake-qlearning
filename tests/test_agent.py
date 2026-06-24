import os

import numpy as np
import pytest
from agent import QLearningAgent


def test_q_table_shape_and_zeros():
    agent = QLearningAgent()
    assert agent.Q.shape == (64, 4)
    assert np.all(agent.Q == 0.0)


def test_choose_action_greedy_when_epsilon_zero():
    agent = QLearningAgent(epsilon=0.0)
    agent.Q[0, 2] = 5.0  # action 2 (Right) is best for state 0
    action = agent.choose_action(0)
    assert action == 2


def test_choose_action_always_valid_range():
    agent = QLearningAgent(epsilon=1.0)
    for _ in range(50):
        action = agent.choose_action(0)
        assert 0 <= action <= 3


def test_update_with_alpha_one_gamma_zero():
    # With alpha=1, gamma=0: Q(s,a) = Q(s,a) + 1*(r + 0 - Q(s,a)) = r
    agent = QLearningAgent(alpha=1.0, gamma=0.0, epsilon=0.0)
    agent.update(0, 2, 1.0, 1, False)
    assert agent.Q[0, 2] == 1.0


def test_update_with_future_value():
    # alpha=1, gamma=0.5, Q[1,*]=0 except Q[1,0]=2.0
    # Q(0,2) = 0 + 1*(0 + 0.5*2.0 - 0) = 1.0
    agent = QLearningAgent(alpha=1.0, gamma=0.5, epsilon=0.0)
    agent.Q[1, 0] = 2.0
    agent.update(0, 2, 0.0, 1, False)
    assert agent.Q[0, 2] == pytest.approx(1.0)


def test_update_done_ignores_next_state():
    # When done=True, target = reward only (no bootstrap)
    # alpha=1, gamma=0.99, Q[1,0]=100 — should be ignored
    agent = QLearningAgent(alpha=1.0, gamma=0.99, epsilon=0.0)
    agent.Q[1, 0] = 100.0
    agent.update(0, 2, 1.0, 1, True)
    assert agent.Q[0, 2] == pytest.approx(1.0)


def test_update_partial_alpha():
    # alpha=0.5, gamma=0, r=1: Q = 0 + 0.5*(1 - 0) = 0.5
    agent = QLearningAgent(alpha=0.5, gamma=0.0, epsilon=0.0)
    agent.update(5, 1, 1.0, 6, False)
    assert agent.Q[5, 1] == pytest.approx(0.5)


def test_decay_epsilon_multiplies():
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.995)


def test_decay_epsilon_respects_floor():
    agent = QLearningAgent(epsilon=0.015, epsilon_min=0.01, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == 0.01


def test_save_and_load(tmp_path):
    agent = QLearningAgent()
    agent.Q[10, 3] = 42.0
    path = str(tmp_path / "q_table.npy")
    agent.save(path)

    agent2 = QLearningAgent()
    agent2.load(path)
    assert agent2.Q[10, 3] == pytest.approx(42.0)
