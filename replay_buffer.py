import random

class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.index = 0

  def push(self, experience):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.index] = experience
    self.index = (self.index + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)