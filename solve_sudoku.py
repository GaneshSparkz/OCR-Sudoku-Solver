import argparse

import cv2
import imutils
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from utils.image_processor import locate_puzzle
from utils.image_processor import extract_digit
from utils.sudoku import Sudoku


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help="path to the trained digit classifier model")
ap.add_argument('-i', '--image', required=True,
                help="path to the image containing the sudoku puzzle")
ap.add_argument('-d', '--debug', type=int, default=0, choices=[0, 1],
                help="set to 1 to visualize each step of the process (default: 0)")
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args['model'])

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)
cv2.imshow("Given Image", image)
cv2.waitKey(0)

# find the puzzle in the image
puzzleImage, warped = locate_puzzle(image, debug=bool(args['debug']))
cv2.imshow("Detected Sudoku Puzzle", puzzleImage)
cv2.waitKey(0)

# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype='int')

# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []

# loop over the grid locations
for y in range(9):
    # initialize the current list of cell locations
    row = []
    for x in range(9):
        # compute the starting and ending (x, y)-coordinates of the
        # current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # crop the cell from the warped transform image and then
        # extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=bool(args['debug']))

        # verify that the digit is not empty
        if digit is not None:
            # resize the cell to 28x28 pixels and then prepare the
            # cell for classification
            roi = cv2.resize(digit, (32, 32))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # classify the digit and update the Sudoku board with the
            # prediction
            pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = pred
            row.append(None)
        else:
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

    cellLocs.append(row)

# construct a Sudoku puzzle from the board
print("[INFO] The recognized Sudoku board...")
puzzle = Sudoku(board.tolist(), 9, 9)
puzzle.print_board()

# solve the Sudoku puzzle
print("[INFO] solving the sudoku puzzle...")
puzzle.solve()
print("[INFO] solved...")
print("[INFO] Sudoku board after solving...")
puzzle.print_board()

# loop over the cell locations and board
for (cellRow, boardRow) in zip(cellLocs, puzzle.board):
    # loop over individual cell in the row
    for (cell, digit) in zip(cellRow, boardRow):
        if cell is None:
            continue

        # unpack the cell coordinates
        startX, startY, endX, endY = cell

        # compute the coordinates of where the digit will be drawn
        # on the output puzzle image
        testX = int((endX - startX) * 0.33)
        testY = int((endY - startY) * -0.2)
        testX += startX
        testY += endY

        # draw the result digit on the Sudoku puzzle image
        cv2.putText(puzzleImage, str(digit), (testX, testY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# show the output image
cv2.imshow("Solved Puzzle", puzzleImage)
cv2.waitKey(0)
cv2.imwrite("solved_puzzle.jpg", puzzleImage)
