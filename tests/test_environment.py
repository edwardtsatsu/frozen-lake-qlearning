import pytest
from environment import FrozenLakeEnv


def test_reset_returns_start_state():
    env = FrozenLakeEnv()
    assert env.reset() == 0


def test_get_state_after_reset():
    env = FrozenLakeEnv()
    env.reset()
    assert env.get_state() == 0


def test_is_terminal_false_for_start():
    env = FrozenLakeEnv()
    env.reset()
    assert env.is_terminal() is False


def test_step_right_from_start():
    env = FrozenLakeEnv()
    env.reset()
    next_state, reward, done = env.step(2)  # Right: (0,0) -> (0,1) = state 1
    assert next_state == 1
    assert reward == 0.0
    assert done is False


def test_step_left_boundary_clamps():
    # Left from col 0 — agent stays in place
    env = FrozenLakeEnv()
    env.reset()  # state 0, (row=0, col=0)
    next_state, reward, done = env.step(0)
    assert next_state == 0
    assert done is False


def test_step_up_boundary_clamps():
    # Up from row 0 — agent stays in place
    env = FrozenLakeEnv()
    env.reset()
    next_state, reward, done = env.step(3)
    assert next_state == 0
    assert done is False


def test_step_right_boundary_clamps():
    # Right from col 7 (state 7) — stays at 7
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 0, 7
    next_state, _, _ = env.step(2)
    assert next_state == 7


def test_step_down_boundary_clamps():
    # Down from row 7 (bottom row) — stays at same state
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 7, 0
    next_state, _, _ = env.step(1)
    assert next_state == 56  # (7,0)


def test_step_into_hole_terminates():
    # (2,2)=18, step Right -> (2,3)=19 which is 'H'
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 2, 2
    next_state, reward, done = env.step(2)
    assert next_state == 19
    assert reward == 0.0
    assert done is True


def test_step_into_goal_gives_reward():
    # (7,6)=62, step Right -> (7,7)=63 which is 'G'
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 7, 6
    next_state, reward, done = env.step(2)
    assert next_state == 63
    assert reward == 1.0
    assert done is True


def test_is_terminal_on_hole():
    # (2,3) = state 19 is a Hole
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 2, 3
    assert env.is_terminal() is True


def test_is_terminal_on_goal():
    # (7,7) = state 63 is the Goal
    env = FrozenLakeEnv()
    env.reset()
    env._row, env._col = 7, 7
    assert env.is_terminal() is True


def test_render_does_not_crash(capsys):
    env = FrozenLakeEnv()
    env.reset()
    env.render()
    captured = capsys.readouterr()
    assert "A" in captured.out  # agent marker appears
