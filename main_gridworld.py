from rl_experiments import RLExperiments
from gridworld_environment import GridworldEnvironment

playground = RLExperiments(problem=GridworldEnvironment)
playground.config_dqn(
    discount_factor=0.98,
    learning_rate=0.001,
    batch_size=128,
    buffer_size=1e6,
    hidden_size=256
)
# experiment_1: compare DQN and DQN with HER on bit flip environment
# playground.experiment_1(problem_size=15, n_epochs=20)
# playground.experiment_1(problem_size=5, n_epochs=5)
# experiment_2: compare the size of bitflip problems that can be solved by DQN and DQN with HER
# playground.experiment_2(max_problem_size=20, max_n_epochs=20, problem_size_step=5)
playground.experiment_2(max_problem_size=12, max_n_epochs=10, problem_size_step=3)
# experiment_3: compare DQN and DQN with HER on bit flip environment for different reward functions
rewards = ["01", "euklidean", "manhattan"]
# playground.experiment_3(problem_size=15, n_epochs=10)
# playground.experiment_3(rewards, problem_size=8, n_epochs=10)