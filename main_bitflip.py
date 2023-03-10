from rl_experiments import RLExperiments
from bitflip_environment import BitFlipEnvironment

playground = RLExperiments(problem=BitFlipEnvironment)
playground.config_dqn(
    discount_factor=0.98,
    learning_rate=0.001,
    batch_size=64,
    buffer_size=512,
    hidden_size=128
)
# experiment_1: compare DQN and DQN with HER on bit flip environment
# playground.experiment_1(problem_size=5, n_epochs=15)
playground.experiment_1(problem_size=6, n_epochs=25)
# experiment_2: compare the size of bitflip problems that can be solved by DQN and DQN with HER
# playground.experiment_2(max_problem_size=20, max_n_epochs=20, problem_size_step=5)
playground.experiment_2(max_problem_size=15, max_n_epochs=10, problem_size_step=3)
# experiment_3: compare DQN and DQN with HER on bit flip environment for different reward functions
rewards = ["01", "mse", "mae"]
# playground.experiment_3(rewards, problem_size=10, n_epochs=10)
playground.experiment_3(rewards, problem_size=4, n_epochs=3)