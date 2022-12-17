"""
a gridworld environment for reinforcement learning
Implemented to test Hindsight Experience Replay.
"""

from typing import Tuple

import torch

from reward_functions import reward_01, reward_euklidean, reward_manhattan
from maze_generator import generate_maze

class GridworldEnvironment:
  def __init__(self, size, reward_type="01", goal_type="random", wall_percentage=0.3, path_eps=0.4):
    """
    Initialize the gridworld environment with a `size` x `size` grid.
    States are the position of the agent in the grid.
    Actions are: 0 = up, 1 = right, 2 = down, 3 = left
    Goals are a random position in the grid that is not the start state.


    Args:
        size: number of rows and columns in the grid
        reward_type: "01" or "euklidean"
    """
    self.size = size
    self.goal_type = goal_type
    self.wall_percentage = wall_percentage
    self.path_eps = path_eps
    if reward_type == "01":
      self.reward_function = reward_01
    elif reward_type.lower() == "euklidean":
      self.reward_function = reward_euklidean
    elif reward_type.lower() == "manhattan":
      self.reward_function = reward_manhattan
    else:
      raise ValueError("Invalid reward type. Valid options are: 01, euklidean, manhattan")
    # initialize maze, state and goal
    self.state: torch.tensor = None
    self.goal: torch.tensor = None
    self.n_goals: int = None
    self.initial_world: torch.tensor = None
    self.world: torch.tensor = None
    self.generate_maze()
    self.reset()


  def get_num_actions(self):
    return 4


  def get_state_size(self):
    return 2

  def generate_maze(self):
    """
    Generate walls in the gridworld for the saved size. Walls are represented by 1.
    """
    maze, start, goal = generate_maze(self.size, self.size, self.wall_percentage, self.path_eps)
    self.initial_world = torch.tensor(maze)
    self.start_state = torch.tensor(start)
    self.goal = torch.tensor(goal)
    # number of potential goals
    self.n_goals = torch.sum(self.initial_world == 0) - 1 # subtract starting position

  def reset(self) -> Tuple[torch.tensor, torch.tensor]:
    """
    Reset the environment to the start state (0,0) and a random goal that is not the start state.

    Returns:
        state: the start state
        goal: a random goal that is not the start state
    """
    # reset state to start position
    self.state = self.start_state.clone()
    self.world = self.initial_world.clone()
    if self.goal_type == "random":
      # choose a random square in world that is not a wall or the start state.
      goal_index = torch.randint(0, self.n_goals, (1,))[0]
      # choose the goal corresponding to the index
      for i in range(self.size):
        for j in range(self.size):
          if self.world[i, j] == 0:
            if goal_index == 0 and (i, j) != (self.state[0], self.state[1]):
              self.goal = torch.tensor([i, j])
              break
            else:
              goal_index -= 1
        else:
          continue
        break

    return self.state.clone(), self.goal.clone()


  def step(self, action: int) -> Tuple[torch.tensor, torch.tensor, bool]:
    """
    Move the agent in the gridworld. If the agent hits a wall or edge of the grid, it stays in the same position.
    Allowed actions are: 0 = up, 1 = right, 2 = down, 3 = left

    Args:
        action: 0 = up, 1 = right, 2 = down, 3 = left
    """
    # get current position
    x, y = self.state
    new_x, new_y = x, y
    # move
    if action == 0: # up
      new_y = min(self.size - 1, y + 1)
    elif action == 1: # right
      new_x = min(self.size - 1, x + 1)
    elif action == 2: # down
      new_y = max(0, y - 1)
    elif action == 3: # left
      new_x = max(0, x - 1)
    else:
      raise ValueError("Invalid action. Valid options are: 0, 1, 2, 3")
    # update state
    if not self.world[new_x, new_y] > 0: # if not wall at new position
      self.state = torch.tensor([new_x, new_y])
    reward, done = self.compute_reward(self.state, self.goal)
    return self.state.clone(), reward, done


  def compute_reward(self, state, goal):
    done = torch.equal(state, goal)
    reward = self.reward_function(state, goal)
    return reward, done

  def __str__(self) -> str:
    """
    return gridworld size
    """
    return f"Gridworld({self.size}x{self.size})"