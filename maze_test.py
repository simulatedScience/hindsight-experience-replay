import numpy as np
from random import shuffle
from typing import List

def generate_maze(gridsize: int) -> np.ndarray:
  """
  generates a simple maze on a square grid of size gridsize by placing walls onto an empty 2D grid.
  Place walls using the following algorithm:
  1. Place a wall at a random position on the grid
  2. If the wall is not connected to the start position, remove the wall
  3. Repeat 1 and 2 until the maze is connected
  4. Repeat 1-3 until the maze is complete

  Args:
      gridsize: the size of the grid

  Returns:
      np.ndarray: a 2D grid with walls represented by 1 and empty space by 0
  """
  # create empty grid
  grid = np.zeros((gridsize, gridsize))
  # create start position
  start = np.array([0,0])
  # create goal position
  goal = np.array([gridsize-1, gridsize-1])
  # create list of possible positions to place walls
  positions = []
  for x in range(gridsize):
    for y in range(gridsize):
      positions.append(np.array([x,y]))
  # shuffle positions
  shuffle(positions)
  # place walls until maze is complete
  while not maze_complete(grid, start, goal):
    # place wall at random position
    wall = positions.pop()
    grid[tuple(wall)] = 1
    # if wall is not connected to start position, remove wall
    if not connected(grid, start, wall):
      grid[tuple(wall)] = 0

  return grid

def maze_complete(grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> bool:
  """
  checks if the maze is complete by checking if the start and goal are connected

  Args:
      grid: the grid with walls represented by 1 and empty space by 0
      start: the start position
      goal: the goal position

  Returns:
      bool: True if the maze is complete, False otherwise
  """
  return connected(grid, start, goal)

def connected(grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> bool:
  """
  checks if the start and goal are connected by checking if there is a path between them

  Args:
      grid: the grid with walls represented by 1 and empty space by 0
      start: the start position
      goal: the goal position

  Returns:
      bool: True if there is a path between the start and goal, False otherwise
  """
  # create list of positions to check
  positions = [start]
  # create list of positions already checked
  checked = []
  # check if goal is connected to start
  while len(positions) > 0:
    # get next position to check
    pos = positions.pop()
    # check if goal is reached
    if np.all(np.equal(pos, goal)):
      return True
    # check if position has already been checked
    if True in [np.all(np.equal(pos, checked_pos)) for checked_pos in checked]:
      continue
    # add position to checked list
    checked.append(pos)
    # get possible moves from position
    moves = get_moves(grid, pos)
    # add possible moves to positions list
    positions.extend(moves)
  # goal is not connected to start
  return False

def get_moves(grid: np.ndarray, pos: np.ndarray) -> List[np.ndarray]:
  """
  gets the possible moves from a position

  Args:
      grid: the grid with walls represented by 1 and empty space by 0
      pos: the current position

  Returns:
      List[np.ndarray]: a list of possible moves
  """
  # get grid size
  gridsize = grid.shape[0]
  # get possible moves
  moves = []
  # check if move up is possible
  if pos[0] > 0 and grid[pos[0]-1, pos[1]] == 0:
    moves.append(np.array([pos[0]-1, pos[1]]))
  # check if move down is possible
  if pos[0] < gridsize-1 and grid[pos[0]+1, pos[1]] == 0:
    moves.append(np.array([pos[0]+1, pos[1]]))
  # check if move left is possible
  if pos[1] > 0 and grid[pos[0], pos[1]-1] == 0:
    moves.append(np.array([pos[0], pos[1]-1]))
  # check if move right is possible
  if pos[1] < gridsize-1 and grid[pos[0], pos[1]+1] == 0:
    moves.append(np.array([pos[0], pos[1]+1]))
  return moves

def print_maze(grid: np.ndarray):
  """
  prints the maze. ⬜ = empty space, ⬛ = wall

  Args:
      grid (np.ndarray): the grid with walls represented by 1 and empty space by 0
  """
  for row in grid:
    for cell in row:
      if cell == 0:
        print("⬜", end="")
      else:
        print("⬛", end="")
    print()

if __name__ == "__main__":
  # generate maze
  grid = generate_maze(10)
  # print maze
  print_maze(grid)