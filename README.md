# OCR Sudoku Solver
Solve the sudoku puzzle from an image using OCR of Digits and Computer Vision

```Shell
usage: train.py [-h] -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to save the trained model file
```

```Shell
usage: solve_sudoku.py [-h] -m MODEL -i IMAGE [-d {0,1}]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to the trained digit classifier model
  -i IMAGE, --image IMAGE
                        path to the image containing the sudoku puzzle
  -d {0,1}, --debug {0,1}
                        set to 1 to visualize each step of the process
                        (default: 0)
```

```Shell
Using TensorFlow backend.
[INFO] loading digit classifier...
[INFO] processing image...
[INFO] The recognized Sudoku board...
+-----------------------+
| 8     |   1   |     9 |
|   5   | 8   7 |   1   |
|     4 |   9   | 7     |
+-----------------------+
|   6   | 7   1 |   2   |
| 5   8 |   6   | 1   7 |
|   1   | 5   2 |   9   |
+-----------------------+
|     7 |   4   | 6     |
|   8   | 3   9 |   4   |
| 3     |   5   |     8 |
+-----------------------+
[INFO] solving the sudoku puzzle...
[INFO] solved...
[INFO] Sudoku board after solving...
+-----------------------+
| 8 7 2 | 4 1 3 | 5 6 9 |
| 9 5 6 | 8 2 7 | 3 1 4 |
| 1 3 4 | 6 9 5 | 7 8 2 |
+-----------------------+
| 4 6 9 | 7 3 1 | 8 2 5 |
| 5 2 8 | 9 6 4 | 1 3 7 |
| 7 1 3 | 5 8 2 | 4 9 6 |
+-----------------------+
| 2 9 7 | 1 4 8 | 6 5 3 |
| 6 8 5 | 3 7 9 | 2 4 1 |
| 3 4 1 | 2 5 6 | 9 7 8 |
+-----------------------+
```
