# Hindsight Experience Replay
This is an implementation of the bit flip experiment in the [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf) paper using Double DQN with a Dueling architecture in PyTorch. This also includes a variety of experiments as well as a gridworld environment to test the algorithm on.

## Requirements
- Python 3.x
- PyTorch
- Numpy
- Matplotlib

## Usage
To run all experiments as I used them in the report, execute `main_combined.py`. Running a single experiment can be done by commentting out the undesired experiments in `main_combined.py` or in `main_bitflip.py` or `main_gridworld.py`.

## Results
Final plots used in the report can be found in the `final_plots` folder. Overall results can be found in the  `B31XS_project_report_SJ.pdf` file.
