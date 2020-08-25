import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


def locate_puzzle(image, debug=False):
    """
    Locates the Sudoku Puzzle in the given image and return the cropped
    puzzle in RGB and Gray scale
    """
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check if we are visualizing to see each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Threshold", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by area in
        # descending order
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleOutline = None

    # loop over the contours
    for contour in contours:
        # approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleOutline = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleOutline is None:
        raise Exception("Could not find Sudoku puzzle outline. "
            "Try debugging your thresholding and contour steps.")
    
    # check if we are visualizing to see the outline of the detected
	# Sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
	    # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleOutline], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    
    # apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
    puzzle = four_point_transform(image, puzzleOutline.reshape(4, 2))
    warped = four_point_transform(gray, puzzleOutline.reshape(4, 2))

    # check if we are visualizing to see the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    
    # return a 2 valued tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)


def extract_digit(cell, debug=False):
    """Extract the digit in the given cell if non-empty"""
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # check if we are visualizing to see the cell thresholding step
    if debug:
        cv2.imshow("Cell Threshold", thresh)
        cv2.waitKey(0)
    
    # find contours in the thresholded cell
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # if no contours were found than this is an empty cell
    if len(contours) == 0:
        return None
    
    # otherwise, find the largest contour in the cell and create a
	# mask for the contour
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    h, w = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # check if we should visualize to see the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    # return the digit to the calling function
    return digit
