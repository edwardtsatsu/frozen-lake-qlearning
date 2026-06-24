# Frozen Lake Q-Learning from First Principles

**DSCD 614 – Reinforcement Learning | University of Ghana | Semester II 2025/2026**

---

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes actions, receives rewards or penalties, and updates its behaviour to maximise cumulative reward over time. Unlike supervised learning, there is no labelled dataset — the agent learns purely from experience.

### What is Frozen Lake?

Frozen Lake is a classic RL benchmark: a grid-world where an agent must navigate from a start position to a goal while avoiding holes in the ice. The 8×8 grid contains four cell types:

| Symbol | Meaning |
|--------|---------|
| `S` | Start position |
| `F` | Frozen (safe) tile |
| `H` | Hole (episode ends, reward 0) |
| `G` | Goal (episode ends, reward +1) |

---

## Environment Design

### State Representation

Each cell is encoded as a flat integer: `state = row * 8 + col`, giving 64 states (0–63). State 0 is the start; state 63 is the goal.

### Action Representation

| Code | Direction | Delta |
|------|-----------|-------|
| 0 | Left  | col − 1 |
| 1 | Down  | row + 1 |
| 2 | Right | col + 1 |
| 3 | Up    | row − 1 |

Moves that would leave the grid are clamped — the agent stays in place.

### Reward Structure

| Event | Reward |
|-------|--------|
| Reach Goal (`G`) | +1.0 |
| Fall in Hole (`H`) | 0.0 |
| Any other step | 0.0 |

Episodes end immediately when the agent reaches `H` or `G`, or after 200 steps (step cap).

---

## Q-Learning Algorithm

### Description

Q-Learning is a model-free, off-policy temporal difference algorithm. It maintains a **Q-table** — a matrix of expected cumulative rewards Q(s, a) for every (state, action) pair. Over many episodes the Q-table converges to the optimal action-value function Q*.

### Update Equation

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

- **α (learning rate):** how much new information overrides old estimates
- **γ (discount factor):** how much future rewards are valued vs immediate
- **r:** reward received after taking action a in state s
- **s':** the next state
- When the episode terminates, the target is just `r` (no bootstrap)

### Exploration Strategy

**ε-greedy with multiplicative decay:**
- With probability ε, pick a random action (explore)
- With probability 1 − ε, pick `argmax Q[state]` (exploit)
- After each episode: `ε ← max(ε_min, ε × ε_decay)`

This transitions from pure exploration early in training to near-pure exploitation as learning matures.

---

## Training Procedure

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning rate α | 0.1 |
| Discount factor γ | 0.99 |
| Initial epsilon ε | 1.0 |
| Min epsilon ε_min | 0.01 |
| Epsilon decay | 0.9997 |
| Max steps per episode | 200 |

### Hyperparameter Experiments

Three configurations were compared (each trained independently for 20,000 episodes):

| Config | α | γ | ε_decay | Notes |
|--------|---|---|---------|-------|
| fast-learn | 0.3 | 0.95 | 0.9995 | Aggressive learning, shorter horizon |
| slow-learn | 0.05 | 0.99 | 0.9997 | Conservative updates |
| **default** | **0.1** | **0.99** | **0.9997** | **Best balance — used for final policy** |

---

## Results

Run `python train.py` to reproduce. Output includes:

- Hyperparameter experiment comparison
- Final success rate
- Learned policy grid
- Training graphs saved to `results/training_results.png`

Run `python evaluate.py` for the 1000-episode greedy evaluation report.

---

## Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent (saves Q-table + graphs)
python train.py

# 3. Evaluate the trained agent
python evaluate.py

# 4. Run unit tests
pytest tests/ -v
```

All output files are saved to `results/`.
