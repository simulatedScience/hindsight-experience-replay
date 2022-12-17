import random
import math
from collections import namedtuple

import torch as torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer

Experience = namedtuple("Experience", field_names="state action reward next_state done")

class DDQNAgent:

  """ Double-DQN with Dueling Architecture """

  def __init__(self,
    state_size: int,
    num_actions: int,
    discount_factor: float = 0.98,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    buffer_size: int = 1e6,
    hidden_size: int = 256):
    # environment parameters
    self.state_size = state_size
    self.num_actions = num_actions
    # hyperparameters for training
    self.discount_factor = discount_factor
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.train_start = max(int(math.sqrt(buffer_size)), batch_size)

    self.memory = ReplayBuffer(int(buffer_size))

    self.Q_network = DuelingMLP(state_size, num_actions, hidden_size)
    self.target_network = DuelingMLP(state_size, num_actions, hidden_size)
    self.update_target_network()

    self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)

  def push_experience(self, state, action, reward, next_state, done):
    self.memory.push(Experience(state, action, reward, next_state, done))

  def update_target_network(self):
    self.target_network.load_state_dict(self.Q_network.state_dict())

  def take_action(self, state, epsilon):
    if random.random() > epsilon:
      return self.greedy_action(state)
    else:
      return torch.randint(self.num_actions, size=())

  def greedy_action(self, state):
    with torch.no_grad():
      return self.Q_network(state).argmax()

  def optimize_model(self):
    if len(self.memory) < self.train_start:
      return

    experiences = self.memory.sample(self.batch_size)
    batch = Experience(*zip(*experiences))

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    non_final_mask = ~torch.tensor(batch.done)
    non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])

    Q_values = self.Q_network(state_batch)[range(self.batch_size), action_batch]

    # Double DQN target #
    next_state_values = torch.zeros(self.batch_size)
    number_of_non_final = sum(non_final_mask)
    with torch.no_grad():
        argmax_actions = self.Q_network(non_final_next_states).argmax(1)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
            range(number_of_non_final), argmax_actions]

    Q_targets = reward_batch + self.discount_factor * next_state_values
    #####################

    assert Q_values.shape == Q_targets.shape

    self.optimizer.zero_grad()
    loss = F.mse_loss(Q_values, Q_targets)
    loss.backward()
    self.optimizer.step()

class DuelingMLP(nn.Module):
  def __init__(self, state_size, num_actions, hidden_size=256):
    super().__init__()
    self.linear = nn.Linear(state_size, hidden_size)
    self.value_head = nn.Linear(hidden_size, 1)
    self.advantage_head = nn.Linear(hidden_size, num_actions)

  def forward(self, x):
    x = x.unsqueeze(0) if len(x.size()) == 1 else x
    x = F.relu(self.linear(x))
    value = self.value_head(x)
    advantage = self.advantage_head(x)
    action_values = (value + (advantage - advantage.mean(dim=1, keepdim=True))).squeeze()
    return action_values