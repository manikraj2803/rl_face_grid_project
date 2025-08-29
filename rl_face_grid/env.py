
import numpy as np
from .utils import gridify, make_target_masks, pick_random_target_cell


class FaceGridEnv:
    """
    A simple Gym-style environment for grid navigation on face images.
    Observation: discrete state index s = r * grid_size + c
    Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    Episode: start at random cell; episode ends on reaching a target cell or max_steps.
    Reward shaping: +1 on target, -step_cost each step, small distance shaping toward target.
    """

    def __init__(self, image, grid_size=16, targets=("eyes", "nose"),
                 max_steps=200, step_cost=0.01, distance_weight=0.1, rng=None):
        self.image = image
        self.grid_size = grid_size
        self.targets = tuple(targets)
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.distance_weight = distance_weight
        self.rng = np.random.default_rng() if rng is None else rng

        H, W, cell_h, cell_w, coords = gridify(image, grid_size)
        self.H, self.W = H, W
        self.cell_h, self.cell_w = cell_h, cell_w
        self.grid_coords = coords
        self.target_masks = make_target_masks(image, grid_size, self.targets)

        self.nS = grid_size * grid_size
        self.nA = 5  # up, down, left, right, stay

        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False
        # pick a start anywhere
        self.r = int(self.rng.integers(0, self.grid_size))
        self.c = int(self.rng.integers(0, self.grid_size))
        # pick a target cell from masks
        self.target_r, self.target_c = pick_random_target_cell(self.target_masks, self.rng)
        return self._state()

    def _state(self):
        return int(self.r * self.grid_size + self.c)

    def _distance(self, r, c):
        # Manhattan distance in grid units
        return abs(r - self.target_r) + abs(c - self.target_c)

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before step().")

        old_dist = self._distance(self.r, self.c)
        # move
        if action == 0 and self.r > 0:  # up
            self.r -= 1
        elif action == 1 and self.r < self.grid_size - 1:  # down
            self.r += 1
        elif action == 2 and self.c > 0:  # left
            self.c -= 1
        elif action == 3 and self.c < self.grid_size - 1:  # right
            self.c += 1
        elif action == 4:
            pass  # stay
        # else: boundary: no movement

        self.steps += 1
        new_dist = self._distance(self.r, self.c)

        # reward shaping
        reward = -self.step_cost
        # distance shaping: positive if moved closer
        reward += self.distance_weight * (old_dist - new_dist)

        # target reward
        if (self.r == self.target_r) and (self.c == self.target_c):
            reward += 1.0
            self.done = True

        # time limit
        if self.steps >= self.max_steps:
            self.done = True

        return self._state(), float(reward), bool(self.done), {
            "position": (self.r, self.c),
            "target": (self.target_r, self.target_c),
            "steps": self.steps,
            "distance": new_dist
        }

    def sample_action(self):
        return int(self.rng.integers(0, self.nA))
