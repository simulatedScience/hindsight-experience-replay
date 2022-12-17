from rl_experiments import RLExperiments
from bitflip_environment import BitFlipEnvironment

playground = RLExperiments(problem=BitFlipEnvironment)
# experiment_1: compare DQN and DQN with HER on bit flip environment
playground.experiment_1(problem_size=15, n_epochs=20)
# playground.experiment_1(n_bits=5, n_epochs=5)
# experiment_2: compare the size of bitflip problems that can be solved by DQN and DQN with HER
playground.experiment_2(max_problem_size=20, max_n_epochs=20, problem_size_step=5)
# experiment_3: compare DQN and DQN with HER on bit flip environment for different reward functions
playground.experiment_3(problem_size=15, n_epochs=10)