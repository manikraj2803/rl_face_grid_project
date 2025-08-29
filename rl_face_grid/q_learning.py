
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, rng=None):
        self.nS = n_states
        self.nA = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng() if rng is None else rng

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.nA))
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = np.max(self.Q[s_next]) if not done else 0.0
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
