"""
a gridworld environment for reinforcement learning
Implemented to test Hindsight Experience Replay.
"""

from typing import Tuple

import torch

class GridworldEnvironment:
  def __init__(self, size, reward_type="01", goal_type="random"):
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
    self.state = torch.tensor([0,0])
    self.goal = torch.tensor([1, 1])
    self.world = torch.zeros((self.size, self.size))
    if reward_type == "01":
      self.reward_function = reward_01
    elif reward_type.lower() == "euklidean":
      self.reward_function = reward_euklidean
    elif reward_type.lower() == "manhattan":
      self.reward_function = reward_manhattan
    else:
      raise ValueError("Invalid reward type. Valid options are: 01, euklidean, manhattan")
    self.initial_world = torch.zeros((self.size, self.size))
    self.generate_walls()
    self.reset()


  def get_num_actions(self):
    return 4


  def get_state_size(self):
    return 2

  def generate_walls(self):
    """
    Generate walls in the gridworld for the saved size. Walls are represented by 1.
    """
    

  def reset(self) -> Tuple[torch.tensor, torch.tensor]:
    """
    Reset the environment to the start state (0,0) and a random goal that is not the start state.

    Returns:
        state: the start state
        goal: a random goal that is not the start state
    """
    # reset state to start position
    self.state = torch.tensor([0,0])
    # choose random square as goal
    if self.goal_type == "random":
      self.goal = torch.randint(self.size, size=(2, ), dtype=torch.float)
      # reset again if state = goal
      if torch.equal(self.state, self.goal):
          self.reset()
    else:
      self.goal = torch.tensor([self.size-1, self.size-1])
    self.world = self.initial_world.clone()
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
    # move
    if action == 0: # up
      new_x = max(0, x - 1)
    elif action == 1: # right
      new_y = min(self.size - 1, y + 1)
    elif action == 2: # down
      new_x = min(self.size - 1, x + 1)
    elif action == 3: # left
      new_y = max(0, y - 1)
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


def reward_01(state: torch.tensor, goal: torch.tensor) -> torch.tensor:
  """
  Compute the 0-1 reward for the state and the goal (1 if state == goal, 0 otherwise).

  Args:
      state (torch.tensor): the current state
      goal (torch.tensor): the goal state

  Returns:
      torch.tensor: 1 if state == goal, 0 otherwise
  """
  return torch.tensor(1.0 if torch.equal(state, goal) else 0.0)

def reward_euklidean(state: torch.tensor, goal: torch.tensor):
  """
  Compute the negative euklidean distance between the state and the goal.

  Args:
      state (torch.tensor): the current state
      goal (torch.tensor): the goal state

  Returns:
      torch.tensor: the euklidean distance between the state and the goal
  """
  return -torch.norm(state - goal)

def reward_manhattan(state: torch.tensor, goal: torch.tensor):
  """
  Compute the negative manhattan distance between the state and the goal.

  Args:
      state (torch.tensor): the current state
      goal (torch.tensor): the goal state

  Returns:
      torch.tensor: the manhattan distance between the state and the goal
  """
  return -torch.sum(torch.abs(state - goal))