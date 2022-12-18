"""
this module implements a class to train NNs with Q-learning for a given problem and perform several experiments.
"""
import os
import random

import torch
import matplotlib.pyplot as plt
# hsv_to_rgb
from matplotlib.colors import hsv_to_rgb

from ddqn_agent import DDQNAgent, Experience
from rl_problem import RLProblem

class RLExperiments:
  def __init__(self, problem: RLProblem):
    self.problem_type = problem
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
      problem: RLProblem,
      max_steps: int = 100,
      num_epochs: int = 10,
      hindsight_replay: bool = True,
      eps_max: float = 0.2,
      eps_min: float = 0.0,
      exploration_fraction: float = 0.7):
    """
    Training loop for the bit flip experiment introduced in https://arxiv.org/pdf/1707.01495.pdf using DQN or DQN with
    hindsight experience replay. 
    The Exploration rate is decayed linearly from `eps_max` to `eps_min` over a fraction of the total
    number of epochs according to the parameter `exploration_fraction`.
    
    Args:
        problem (RLProblem): Problem to solve (must already be initialized)
        max_steps (int): Maximum number of steps (= actions) per episode.
        num_epochs (int): Number of epochs to train for.
        hindsight_replay (bool): Whether to use hindsight experience replay or not.
        eps_max (float): Maximum exploration rate.
        eps_min (float): Minimum exploration rate.
        exploration_fraction (float): Fraction of the total number of epochs over which to decay the exploration rate
            from eps_max to eps_min.
        reward_type (str): Reward function to use, defaults to "01". Possible reward functions depend on the problem.

    Returns:
        (list): List of success rates over the epochs.
    """
    # Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
    future_k = 4
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40

    state_size = problem.get_state_size()
    num_actions = problem.get_num_actions()
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
          state, goal = problem.reset()

          for step in range(max_steps):
            state_ = torch.cat((state, goal))
            action = agent.take_action(state_, eps)
            next_state, reward, done = problem.step(action.item())
            episode_trajectory.append(Experience(state, action, reward, next_state, done))
            state = next_state
            if done:
              successes += 1
              break

          # Fill up replay memory
          steps_taken = step
          for t in range(steps_taken):
            # Standard experience replay
            state, action, reward, next_state, done = episode_trajectory[t]
            state_, next_state_ = torch.cat((state, goal)), torch.cat((next_state, goal))
            agent.push_experience(state_, action, reward, next_state_, done)

            # Hindsight experience replay with future strategy
            if hindsight_replay:
              for _ in range(future_k):
                future = random.randint(t, steps_taken)  # index of future time step
                new_goal = episode_trajectory[future].next_state  # take future next_state and set as goal
                new_reward, new_done = problem.compute_reward(next_state, new_goal)
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
    problem = self.problem_type(problem_size, "01")
    for her in [True, False]:
      label = "HER" if her else "DQN"
      print(f"start training with {label}")
      success_rates = self.train(problem, problem.get_max_steps(), n_epochs, her)
      label += f" (01)"
      plt.plot(
          range(1, len(success_rates) + 1),
          success_rates,
          linestyle="-",
          label=label)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title(f"Exp. 1: HER for DQN - {problem}")
    plt.grid(color="#dddddd")
    filename = f"{problem}_exp1_{n_epochs}_epochs.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.clf() # clear plot

  def experiment_2(self, max_problem_size=30, max_n_epochs=30, problem_size_step=5):
    """
    experiment_2 compares the size of bitflip problems that can be solved by DQN and DQN with hindsight experience replay.

    Args:
        max_problem_size (int, optional): maximum size of bitflip problem. Defaults to 50.
        max_n_epochs (int, optional): maximum number of epochs to train for. Defaults to 50.
    """
    print("starting experiment 2")
    problem_sizes = list(range(problem_size_step, max_problem_size + 1, problem_size_step))
    final_success_rates = [[], []]
    for problem_size in problem_sizes:
      problem = self.problem_type(problem_size, "01")
      for her in [True, False]:
        label = "HER" if her else "DQN"
        print(f"start training with {label}, {problem_size} bits")
        success_rates = self.train(problem, problem.get_max_steps(), max_n_epochs, her)
        final_success_rates[her].append(success_rates[-1]) # True = 1, False = 0
    plt.plot(problem_sizes, final_success_rates[1], linestyle="-", label="HER")
    plt.plot(problem_sizes, final_success_rates[0], linestyle="-", label="DQN")
    plt.legend()
    plt.xlabel("Number of bits")
    plt.ylabel("Final success rate")
    plt.title(f"Exp. 2: Final success HER for DQN - {problem}, {max_n_epochs} epochs")
    plt.grid(color="#dddddd")
    filename = f"{problem}_exp2_{max_n_epochs}_epochs_{problem_size_step}_step.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.clf() # clear plot


  def experiment_3(self, rewards, problem_size=25, n_epochs=25):
    """
    experiment_3 evaluates the performance of DQN with hindsight experience replay on the bit flip environment for different reward functions.

    Args:
        rewards (list): list of reward functions to evaluate.
        problem_size (int, optional): _description_. Defaults to 50.
        n_epochs (int, optional): _description_. Defaults to 50.
    """
    linestyles = ["-", "--", "-.", ":"]*(len(rewards)//4 + 1)
    print("starting experiment 3")
    for i, (reward, linestyle) in enumerate(zip(rewards, linestyles)):
      problem = self.problem_type(problem_size, reward)
      for her in [True, False]:
        label = "HER" if her else "DQN"
        print(f"start training with {label}, {reward}-reward")
        success_rates = self.train(problem, problem.get_max_steps(), n_epochs, her)
        label += f" ({reward})"
        plt.plot(
            range(1, len(success_rates) + 1),
            success_rates,
            linestyle=linestyle,
            color=get_color(her, i),
            label=label)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title(f"Exp. 3: HER and reward shaping - {problem}")
    plt.grid(color="#dddddd")
    filename = f"{problem}_exp3_{n_epochs}_epochs.png"
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.clf() # clear plot

def get_color(her, reward_index):
  if her: # blue tone depending on reward index
    hue = 222
  else: # orange tone depending on reward index
    hue = 30
  sat = 60 + 20 * reward_index
  val = 100 - 20 * reward_index
  # return color for matplotlib
  return hsv_to_rgb((hue/360, sat/100, val/100))