from rl_experiments import RLExperiments
from bitflip_environment import BitFlipEnvironment
from gridworld_environment import GridworldEnvironment

def main_bitflip():
  bit_playground = RLExperiments(problem=BitFlipEnvironment)
  bit_playground.config_dqn(
      discount_factor=0.98,
      learning_rate=0.001,
      batch_size=128,
      buffer_size=2048,
      hidden_size=256
  )
  # experiment_1: compare DQN and DQN with HER on bit flip environment
  bit_playground.experiment_1(problem_size=15, n_epochs=25)
  # experiment_2: compare the size of bitflip problems that can be solved by DQN and DQN with HER
  bit_playground.experiment_2(max_problem_size=21, max_n_epochs=20, problem_size_step=3)
  # experiment_3: compare DQN and DQN with HER on bit flip environment for different reward functions
  rewards = ["01", "mse", "mae"]
  bit_playground.experiment_3(rewards, problem_size=15, n_epochs=20)


def main_gridworld():
  grid_playground = RLExperiments(problem=GridworldEnvironment)
  grid_playground.config_dqn(
      discount_factor=0.98,
      learning_rate=0.001,
      batch_size=128,
      buffer_size=2048,
      hidden_size=256
  )
  # experiment_1: compare DQN and DQN with HER on bit flip environment
  grid_playground.experiment_1(problem_size=15, n_epochs=20)
  # experiment_2: compare the size of bitflip problems that can be solved by DQN and DQN with HER
  grid_playground.experiment_2(max_problem_size=21, max_n_epochs=15, problem_size_step=3)
  # experiment_3: compare DQN and DQN with HER on bit flip environment for different reward functions
  rewards = ["01", "euclidean", "manhattan"]
  grid_playground.experiment_3(rewards, problem_size=12, n_epochs=15)

if __name__ == "__main__":
  main_bitflip()
  main_gridworld()