import numpy as np
from dataclasses import dataclass
import random

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict


class GridWorldEnv:
    """
    GridWorld environment with Snake-style continuous training.
    - 10×10 grid
    - Goal respawns after being reached (like food in Snake)
    - Obstacles respawn with goal (dynamic environment)
    - Bomb stays fixed until episode ends (permanent hazard)
    - Episode only ends on collision or timeout
    - Distance-based reward shaping (like Snake)
    """

    def __init__(self, grid_size=10, max_steps=1000, use_obstacles=True, fixed_start=True):
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
            cell = (random.randint(0, self.n - 1), random.randint(0, self.n - 1))
            if cell not in forbidden:
                return cell

    def _get_distance_to_goal(self):
        """Manhattan distance between agent and goal."""
        return abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])

    def _get_distance_to_bomb(self):
        """Manhattan distance between agent and bomb."""
        return abs(self.agent[0] - self.bomb[0]) + abs(self.agent[1] - self.bomb[1])

    def _get_min_distance_to_obstacles(self):
        """Minimum Manhattan distance to any obstacle."""
        if not self.obstacles:
            return float('inf')
        distances = [abs(self.agent[0] - obs[0]) + abs(self.agent[1] - obs[1]) 
                     for obs in self.obstacles]
        return min(distances)

    def reset(self):
        """Reset environment to initial state (called when episode ends)."""
        self.t = 0
        self.score = 0
        self.steps_without_goal = 0
        forbidden = set()

        # Agent position (fixed or random start)
        self.agent = (0, 0) if self.fixed_start else self._rand_free_cell(forbidden)
        forbidden.add(self.agent)

        # Bomb (respawns on episode reset only)
        self.bomb = self._rand_free_cell(forbidden)
        forbidden.add(self.bomb)

        # Goal (will respawn when reached)
        self.goal = self._rand_free_cell(forbidden)
        forbidden.add(self.goal)

        # Obstacles (will respawn with goal)
        self.obstacles = set()
        if self.use_obstacles:
            for _ in range(2):
                cell = self._rand_free_cell(forbidden)
                self.obstacles.add(cell)
                forbidden.add(cell)

        self.done = False
        self.prev_goal_distance = self._get_distance_to_goal()
        self.prev_bomb_distance = self._get_distance_to_bomb()
        self.prev_obstacle_distance = self._get_min_distance_to_obstacles()
        
        return self._get_obs()

    def _respawn_goal_and_obstacles(self):
        """
        Respawn goal and obstacles at new locations (like food respawning in Snake).
        Bomb stays at the same location.
        """
        forbidden = {self.agent, self.bomb}
        
        # Respawn goal
        self.goal = self._rand_free_cell(forbidden)
        forbidden.add(self.goal)
        
        # Respawn obstacles at new locations
        self.obstacles = set()
        if self.use_obstacles:
            for _ in range(2):
                cell = self._rand_free_cell(forbidden)
                self.obstacles.add(cell)
                forbidden.add(cell)
        
        # Reset distance tracking
        self.prev_goal_distance = self._get_distance_to_goal()
        self.prev_obstacle_distance = self._get_min_distance_to_obstacles()

    def _get_obs(self):
        """
        Return 20-dim binary feature vector:
        
        Danger detection in 8 directions (8 binary features):
        - danger_up, danger_down, danger_left, danger_right
        - danger_up_left, danger_up_right, danger_down_left, danger_down_right
        
        Goal relative location (4 binary features):
        - goal_up, goal_down, goal_left, goal_right
        
        Bomb relative location (4 binary features):
        - bomb_up, bomb_down, bomb_left, bomb_right
        
        Nearest obstacle relative location (4 binary features):
        - obstacle_up, obstacle_down, obstacle_left, obstacle_right
        """
        ai, aj = self.agent
        gi, gj = self.goal
        bi, bj = self.bomb

        def blocked(cell):
            """Check if cell is dangerous (bomb, obstacle, or wall)."""
            return (cell == self.bomb or cell in self.obstacles or 
                    not (0 <= cell[0] < self.n and 0 <= cell[1] < self.n))

        # Danger detection in 8 directions (like Snake)
        danger_up = int(blocked((ai - 1, aj)))
        danger_down = int(blocked((ai + 1, aj)))
        danger_left = int(blocked((ai, aj - 1)))
        danger_right = int(blocked((ai, aj + 1)))
        danger_up_left = int(blocked((ai - 1, aj - 1)))
        danger_up_right = int(blocked((ai - 1, aj + 1)))
        danger_down_left = int(blocked((ai + 1, aj - 1)))
        danger_down_right = int(blocked((ai + 1, aj + 1)))

        # Goal relative location
        goal_up = int(gi < ai)
        goal_down = int(gi > ai)
        goal_left = int(gj < aj)
        goal_right = int(gj > aj)

        # Bomb relative location
        bomb_up = int(bi < ai)
        bomb_down = int(bi > ai)
        bomb_left = int(bj < aj)
        bomb_right = int(bj > aj)

        # Nearest obstacle relative location
        if self.obstacles:
            min_dist = float('inf')
            nearest_obs = None
            for obs in self.obstacles:
                dist = abs(ai - obs[0]) + abs(aj - obs[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = obs
            
            obs_i, obs_j = nearest_obs
            obstacle_up = int(obs_i < ai)
            obstacle_down = int(obs_i > ai)
            obstacle_left = int(obs_j < aj)
            obstacle_right = int(obs_j > aj)
        else:
            obstacle_up = obstacle_down = obstacle_left = obstacle_right = 0

        return np.array([
            # 8 danger features
            danger_up, danger_down, danger_left, danger_right,
            danger_up_left, danger_up_right, danger_down_left, danger_down_right,
            
            # 4 goal direction features
            goal_up, goal_down, goal_left, goal_right,
            
            # 4 bomb direction features
            bomb_up, bomb_down, bomb_left, bomb_right,
            
            # 4 nearest obstacle direction features
            obstacle_up, obstacle_down, obstacle_left, obstacle_right
        ], dtype=np.float32)
    
    def set_goal(self, row, col):
        """Manually set goal position and update distances."""
        if (row, col) not in {self.agent, self.bomb} and (row, col) not in self.obstacles:
            self.goal = (row, col)
            self.prev_goal_distance = self._get_distance_to_goal()


    def step(self, action):
        """
        Execute one step in the environment.
        Snake-style rewards and continuous episodes.
        """
        self.t += 1
        self.steps_without_goal += 1
        
        di, dj = self.ACTIONS[action]
        ni, nj = self.agent[0] + di, self.agent[1] + dj

        # Check wall collision (episode ends)
        if not (0 <= ni < self.n and 0 <= nj < self.n):
            return StepResult(self._get_obs(), -10.0, True, {"reason": "wall"})

        # Move agent
        self.agent = (ni, nj)

        # Check obstacle collision (episode ends)
        if self.agent in self.obstacles:
            return StepResult(self._get_obs(), -10.0, True, {"reason": "obstacle"})

        # Check bomb collision (episode ends)
        if self.agent == self.bomb:
            return StepResult(self._get_obs(), -10.0, True, {"reason": "bomb"})

        # Initialize reward
        reward = 0.0

        # Check goal reached (episode CONTINUES - like eating food in Snake)
        if self.agent == self.goal:
            self.score += 1
            reward = 12.0  # Same as Snake food reward
            self.steps_without_goal = 0
            
            # Respawn goal and obstacles at new locations
            self._respawn_goal_and_obstacles()
            
            return StepResult(self._get_obs(), reward, False, {"reason": "goal"})

        # Distance-based reward shaping (like Snake)
        current_goal_distance = self._get_distance_to_goal()
        current_bomb_distance = self._get_distance_to_bomb()
        current_obstacle_distance = self._get_min_distance_to_obstacles()

        # GOAL: Reward for moving closer, penalty for moving away
        # (Balanced rewards to discourage oscillation)
        if current_goal_distance < self.prev_goal_distance:
            reward += 2.0  # Moving closer to goal (POSITIVE)
        else:
            reward -= 2.0  # Moving away from goal (NEGATIVE)

        # BOMB: Penalty for moving closer, reward for moving away
        if current_bomb_distance < self.prev_bomb_distance:
            reward -= 0.5  # Getting closer to bomb is dangerous (NEGATIVE)
        else:
            reward += 0.3  # Moving away from bomb is safe (POSITIVE)
        
        # OBSTACLES: Penalty for moving closer, reward for moving away
        if current_obstacle_distance < self.prev_obstacle_distance:
            reward -= 0.4  # Getting closer to obstacles is dangerous (NEGATIVE)
        else:
            reward += 0.2  # Moving away from obstacles is safe (POSITIVE)

        # NO survival bonus - forces agent to make progress toward goal

        # Update distance tracking
        self.prev_goal_distance = current_goal_distance
        self.prev_bomb_distance = current_bomb_distance
        self.prev_obstacle_distance = current_obstacle_distance

        # Check timeout (episode ends)
        if self.steps_without_goal >= self.max_steps:
            return StepResult(self._get_obs(), -10.0, True, {"reason": "timeout"})

        # Normal step (episode continues)
        return StepResult(self._get_obs(), reward, False, {})

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
        print(f"Score: {self.score} | Step: {self.t} | "
              f"Steps without goal: {self.steps_without_goal}/{self.max_steps}\n")