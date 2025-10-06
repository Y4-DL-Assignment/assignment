import numpy as np
from dataclasses import dataclass

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict


class GridWorldEnv:
    """
    GridWorld environment with obstacles, goal, and bomb.
    Feature-based 12-dimensional state for DQN.
    """

    def __init__(self, grid_size=6, max_steps=200, use_obstacles=True, fixed_start=False):
        self.n = grid_size
        self.max_steps = max_steps
        self.use_obstacles = use_obstacles
        self.fixed_start = fixed_start
        self.ACTIONS = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }
        self.obstacles = set()
        self.reset()

    def _rand_free_cell(self, forbidden):
        """Generate random free cell not in forbidden set."""
        while True:
            cell = (np.random.randint(self.n), np.random.randint(self.n))
            if cell not in forbidden:
                return cell

    def reset(self):
        """Reset environment to initial state."""
        self.t = 0
        self.score = 0
        forbidden = set()

        # Agent position
        self.agent = (0, 0) if self.fixed_start else self._rand_free_cell(forbidden)
        forbidden.add(self.agent)

        # Goal and bomb
        self.goal = self._rand_free_cell(forbidden)
        forbidden.add(self.goal)
        self.bomb = self._rand_free_cell(forbidden)
        forbidden.add(self.bomb)

        # Obstacles
        self.obstacles = set()
        if self.use_obstacles:
            for _ in range(2):
                cell = self._rand_free_cell(forbidden)
                self.obstacles.add(cell)
                forbidden.add(cell)

        self.done = False
        return self._get_obs()

    def _get_obs(self):
        """
        Return 12-dim feature vector:
        [danger_up, danger_down, danger_left, danger_right,
         goal_up, goal_down, goal_left, goal_right,
         bomb_up, bomb_down, bomb_left, bomb_right]
        """
        ai, aj = self.agent
        gi, gj = self.goal
        bi, bj = self.bomb

        def blocked(cell):
            return (cell == self.bomb or cell in self.obstacles or 
                    not (0 <= cell[0] < self.n and 0 <= cell[1] < self.n))

        return np.array([
            int(blocked((ai - 1, aj))),  # danger_up
            int(blocked((ai + 1, aj))),  # danger_down
            int(blocked((ai, aj - 1))),  # danger_left
            int(blocked((ai, aj + 1))),  # danger_right
            int(gi < ai),  # goal_up
            int(gi > ai),  # goal_down
            int(gj < aj),  # goal_left
            int(gj > aj),  # goal_right
            int(bi < ai),  # bomb_up
            int(bi > ai),  # bomb_down
            int(bj < aj),  # bomb_left
            int(bj > aj),  # bomb_right
        ], dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""
        self.t += 1
        di, dj = self.ACTIONS[action]
        ni, nj = self.agent[0] + di, self.agent[1] + dj

        # Check wall collision
        if not (0 <= ni < self.n and 0 <= nj < self.n):
            return StepResult(self._get_obs(), -10.0, True, {"reason": "wall"})

        # Move agent
        self.agent = (ni, nj)

        # Check obstacle collision
        if self.agent in self.obstacles:
            return StepResult(self._get_obs(), -10.0, True, {"reason": "obstacle"})

        # Check bomb collision
        if self.agent == self.bomb:
            return StepResult(self._get_obs(), -10.0, True, {"reason": "bomb"})

        # Check goal reached
        if self.agent == self.goal:
            self.score += 1
            return StepResult(self._get_obs(), 100.0, True, {"reason": "goal"})

        # Check timeout
        if self.t >= self.max_steps:
            return StepResult(self._get_obs(), -1.0, True, {"reason": "timeout"})

        # Normal step
        return StepResult(self._get_obs(), -0.1, False, {})

    def add_obstacle(self, pos):
        """Add obstacle at position (max 2)."""
        if len(self.obstacles) < 2 and pos not in {self.goal, self.bomb, self.agent}:
            self.obstacles.add(pos)
            return True
        return False

    def remove_obstacle(self, pos):
        """Remove obstacle at position."""
        if pos in self.obstacles:
            self.obstacles.remove(pos)
            return True
        return False

    def render(self):
        """Print text representation of grid."""
        grid = np.full((self.n, self.n), ".", dtype=object)
        ai, aj = self.agent
        gi, gj = self.goal
        bi, bj = self.bomb
        for oi, oj in self.obstacles:
            grid[oi, oj] = "#"
        grid[gi, gj] = "G"
        grid[bi, bj] = "B"
        grid[ai, aj] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print(f"Score: {self.score} | Step: {self.t}\n")