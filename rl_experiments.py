"""
this module implements a class to train NNs with Q-learning for a given problem and perform several experiments.
"""
import os
import random

import torch
import matplotlib.pyplot as plt

from ddqn_agent import DDQNAgent, Experience

class RLExperiments:
  def __init__(self, problem):
    self.problem_type = problem
    self.problem: self.problem_type = None
    self.dqn_params = {
      "discount_factor": 0.98,
      "learning_rate": 0.001,
      "batch_size": 128,
      "buffer_size": 1e6,
      "hidden_size": 256
    }

  def config_dqn(self,
    discount_factor: float = 0.98,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    buffer_size: int = 1e6,
    hidden_size: int = 256):
    """
    save parameters for DQN agent to be used in all future experiments (or until this method is called again).
    """
    self.dqn_params = {
      "discount_factor": discount_factor,
      "learning_rate": learning_rate,
      "batch_size": batch_size,
      "buffer_size": buffer_size,
      "hidden_size": hidden_size
    }

  def train(self,
      problem_size=10,
      num_epochs=10,
      hindsight_replay=True,
      eps_max=0.2,
      eps_min=0.0,
      exploration_fraction=0.7,
      reward_type="01"):
    """
    Training loop for the bit flip experiment introduced in https://arxiv.org/pdf/1707.01495.pdf using DQN or DQN with
    hindsight experience replay. 
    The Exploration rate is decayed linearly from `eps_max` to `eps_min` over a fraction of the total
    number of epochs according to the parameter `exploration_fraction`.
    
    Args:
        problem_size (int): Number of bits in the bit flip environment.
        num_epochs (int): Number of epochs to train for.
        hindsight_replay (bool): Whether to use hindsight experience replay or not.
        eps_max (float): Maximum exploration rate.
        eps_min (float): Minimum exploration rate.
        exploration_fraction (float): Fraction of the total number of epochs over which to decay the exploration rate
            from eps_max to eps_min.
        reward_type (str): Reward function to use. Can be either "01", "mse" or "mae".

    Returns:
        (list): List of success rates over the epochs.
    """
    # initialize problem
    self.problem = self.problem_type(problem_size, reward_type)

    # Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
    future_k = 4
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40

    state_size = self.problem.get_state_size()
    num_actions = self.problem.get_num_actions()
    agent = DDQNAgent(
        state_size, 
        num_actions,
        **self.dqn_params)

    success_rate = 0.0
    success_rates = []
    for epoch in range(num_epochs):

      # Decay epsilon linearly from eps_max to eps_min
      eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)

      successes = 0
      for cycle in range(num_cycles):
        for episode in range(num_episodes):
          # Run episode and cache trajectory
          episode_trajectory = []
          state, goal = self.problem.reset()

          for step in range(problem_size):
            state_ = torch.cat((state, goal))
            action = agent.take_action(state_, eps)
            next_state, reward_type, done = self.problem.step(action.item())
            episode_trajectory.append(Experience(state, action, reward_type, next_state, done))
            state = next_state
            if done:
              successes += 1
              break

          # Fill up replay memory
          steps_taken = step
          for t in range(steps_taken):
            # Standard experience replay
            state, action, reward_type, next_state, done = episode_trajectory[t]
            state_, next_state_ = torch.cat((state, goal)), torch.cat((next_state, goal))
            agent.push_experience(state_, action, reward_type, next_state_, done)

            # Hindsight experience replay with future strategy
            if hindsight_replay:
              for _ in range(future_k):
                future = random.randint(t, steps_taken)  # index of future time step
                new_goal = episode_trajectory[future].next_state  # take future next_state and set as goal
                new_reward, new_done = self.problem.compute_reward(next_state, new_goal)
                state_, next_state_ = torch.cat((state, new_goal)), torch.cat((next_state, new_goal))
                agent.push_experience(state_, action, new_reward, next_state_, new_done)

        # Optimize DQN
        for opt_step in range(num_opt_steps):
            agent.optimize_model()

        agent.update_target_network()

      success_rate = successes / (num_episodes * num_cycles)
      success_rates.append(success_rate)
      print(f"Epoch: {epoch + 1}, exploration: {100 * eps:.0f}%, success rate: {success_rate:.2f}")
      if success_rate > 0.995:
        print("Ending training early, success rate is above 99.5%")
        break

    return success_rates


  def experiment_1(self, problem_size=30, n_epochs=30):
    """
    experiment_1 compares the performance of DQN and DQN with hindsight experience replay on the bit flip environment.

    Args:
        problem_size (int, optional): Number of bits in the bit flip environment. Defaults to 30.
        n_epochs (int, optional): Number of epochs to train for. Defaults to 30.
    """
    print("starting experiment 1")
    for her in [True, False]:
      label = "HER" if her else "DQN"
      print(f"start training with {label}")
      success = self.train(problem_size, n_epochs, her)
      label += f" (01)"
      plt.plot(success, linestyle="-", label=label)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title(f"HER for DQN - {problem_size} bits")
    plt.grid(color="#dddddd")
    filename = f"{self.problem}_exp1_{problem_size}_bits_{n_epochs}_epochs.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.show()

  def experiment_2(self, max_problem_size=30, max_n_epochs=30, problem_size_step=5):
    """
    experiment_2 compares the size of bitflip problems that can be solved by DQN and DQN with hindsight experience replay.

    Args:
        max_problem_size (int, optional): maximum size of bitflip problem. Defaults to 50.
        max_n_epochs (int, optional): maximum number of epochs to train for. Defaults to 50.
    """
    print("starting experiment 2")
    for her in [True, False]:
      final_success_rates = []
      label = "HER" if her else "DQN"
      for problem_size in range(problem_size_step, max_problem_size + 1, problem_size_step):
        print(f"start training with {label}, {problem_size} bits")
        success = self.train(problem_size, max_n_epochs, her)
        final_success_rates.append(success[-1])
      plt.plot(final_success_rates, linestyle="-", label=label)
    plt.legend()
    plt.xlabel("Number of bits")
    plt.ylabel("Final success rate")
    plt.title(f"Final success HER for DQN - {max_n_epochs} epochs")
    plt.grid(color="#dddddd")
    filename = f"exp2_{max_problem_size}_bits_{max_n_epochs}_epochs.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.show()


  def experiment_3(self, problem_size=25, n_epochs=25):
    """
    experiment_3 evaluates the performance of DQN with hindsight experience replay on the bit flip environment for different reward functions.

    Args:
        problem_size (int, optional): _description_. Defaults to 50.
        n_epochs (int, optional): _description_. Defaults to 50.
    """
    print("starting experiment 3")
    for reward, linestyle in zip(["01", "mse"], ["-", "--"]):
      for her in [True, False]:
        label = "HER" if her else "DQN"
        print(f"start training with {label}, {reward}-reward")
        success = self.train(problem_size, n_epochs, her, reward_type=reward)
        label += f" ({reward})"
        plt.plot(success, linestyle=linestyle, label=label)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title(f"HER for DQN - {problem_size} bits")
    plt.grid(color="#dddddd")
    filename = f"exp3_{problem_size}_bits_{n_epochs}_epochs.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.show()
