"""Tabular Q-learning (a Temporal-Difference method) for CartPole with a discretised state."""
import numpy as np

BINS = (3, 3, 8, 8)
LOW = np.array([-2.4, -3.0, -0.21, -3.5])
HIGH = np.array([2.4, 3.0, 0.21, 3.5])


def discretize(obs):
    ratios = (np.clip(obs, LOW, HIGH - 1e-6) - LOW) / (HIGH - LOW)
    return tuple((ratios * np.array(BINS)).astype(int))


class QAgent:
    def __init__(self, n_actions=2, alpha=0.2, gamma=0.99, seed=0):
        self.q = np.zeros(BINS + (n_actions,))
        self.alpha, self.gamma = alpha, gamma
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[s]))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * np.max(self.q[s2])
        self.q[s + (a,)] += self.alpha * (target - self.q[s + (a,)])
