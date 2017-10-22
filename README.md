# Lizard-problem

This is a variation of the N-queens problem where we have to place `m` lizards on a `nxn` board. A lizard can attack another lizard on the same row, column or diagnol. We have to ensure that the placement is such that no lizard can attack another lizard. The extension is that the board can also have trees that may prevent a lizard from attacking another lizard on the same row, column or diagnol. This allows us to place more than `n` lizards on a `nxn` board.

## Techniques Used

- BFS
- DFS
- Simulated Annealing

## Input file

The format of an input file is as follows

- First line: instruction of which algorithm to use: BFS, DFS or SA
- Second line: strictly positive 32-bit integer N, the width and height of the square nursery
- Third line:  strictly positive 32-bit integer p, the number of queens
- Next N lines: the N x N board, one file line per row (to show you where the walls are). It will have a 0 where there is nothing, and a 2 where there is a wall.

## Output file

The output will be printed in a file named `output.txt` 

- First line: OK or FAIL, indicating whether a solution was found or not. If FAIL, any following lines are ignored.
- Next N lines: the n x n nursery, one line in the file per nursery row, including the baby lizards and trees. It will have a 0 where there is nothing, a 1 where you placed a baby lizard, and a 2 where there is a tree.

## Running the code

- Place your file `input.txt` in the same folder as the code
- Run the code using `python homework.py`
