"""
a bitflip environment for reinforcement learning
Implemented to test Hindsight Experience Replay.
"""

from typing import Tuple

import torch

class BitFlipEnvironment:
  def __init__(self, n_bits, reward_type="01"):
    """
    Initialize the bit flip environment with `n_bits` bits.

    Args:
        n_bits: number of bits in the bit string
        reward_type: "01" or "hamming"
    """
    self.n_bits = n_bits
    self.state = torch.zeros((self.n_bits, ))
    self.goal = torch.zeros((self.n_bits, ))
    if reward_type == "01":
      self.reward_function = reward_01
    elif reward_type.lower() == "mse":
      self.reward_function = reward_mse
    elif reward_type.lower() == "mae":
      self.reward_function = reward_mae
    else:
      raise ValueError("Invalid reward type. Valid options are: 01, mse, mae")
    self.reset()

  def get_num_actions(self):
    return self.n_bits

  def get_state_size(self):
    return self.n_bits*2

  def reset(self):
    self.state = torch.randint(2, size=(self.n_bits, ), dtype=torch.float)
    self.goal = torch.randint(2, size=(self.n_bits, ), dtype=torch.float)
    # reset again if state = goal
    if torch.equal(self.state, self.goal):
        self.reset()
    return self.state.clone(), self.goal.clone()


  def step(self, action) -> Tuple[torch.tensor, torch.tensor, bool]:
    self.state[action] = 1 - self.state[action]  # Flip the bit on position of the action
    reward, done = self.compute_reward(self.state, self.goal)
    return self.state.clone(), reward, done


  def render(self):
    print("State: {}".format(self.state.tolist()))
    print("Goal : {}\n".format(self.goal.tolist()))


  def compute_reward(self, state: torch.tensor, goal: torch.tensor):
    done = torch.equal(state, goal)
    reward = self.reward_function(state, goal)
    # reward = torch.tensor(1.0 if done else 0.0)
    return reward, done

  def __str__(self):
    return f"BitFlipEnvironment(n_bits={self.n_bits})"



def reward_01(state: torch.tensor, goal: torch.tensor):
  """
  Reward 1 for success (state == goal), 0 otherwise.

  Args:
      state (torch.tensor): Current state.
      goal (torch.tensor): Goal state.

  Returns:
      (torch.tensor): Reward
      (bool): True if state == goal, False otherwise.
  """
  done = torch.equal(state, goal)
  return torch.tensor(1.0 if done else 0.0)


def reward_mse(state: torch.tensor, goal: torch.tensor):
  """
  Reward is the negative mean squared error between state and goal.

  Args:
      state (torch.tensor): Current state.
      goal (torch.tensor): Goal state.

  Returns:
      (torch.tensor): Reward
  """
  return -torch.norm(state - goal, p=2)


def reward_mae(state: torch.tensor, goal: torch.tensor):
  """
  Reward is the negative mean absolute error between state and goal.

  Args:
      state (torch.tensor): Current state.
      goal (torch.tensor): Goal state.

  Returns:
      (torch.tensor): Reward
  """
  return -torch.norm(state - goal, p=1)