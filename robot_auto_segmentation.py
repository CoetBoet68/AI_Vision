import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
# from tkinter import *
# from PIL import Image, ImageTk

image = cv.imread("imgs/robo_table.png")
assert image is not None, "file could not be read, check with os.path.exists()"

cv.imshow("Original image", image)
cv.waitKey(0)

cropped_image = image[220:430, 340:430] # [220:420, 340:430] for conveyor belt 

# Save the cropped image
cv.imwrite("imgs/cropped_robot.png", cropped_image)

cv.imshow("Cropped image", cropped_image)
cv.waitKey(0)

gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

cv.imshow("Gray image", gray_image)
cv.waitKey(0)

# blur_image = cv.medianBlur(gray_image, 3)

thresh_img = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

cv.imshow("_", thresh_img)
cv.waitKey(0)

# noise removal
kernel = np.ones((3,3),np.uint8)

# Opening is just another name of erosion followed by dilation
opening_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel, iterations = 2)
cv.imshow("Opening", opening_img)
cv.waitKey(0)

edges = cv.Canny(opening_img, 30, 200)

contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(cropped_image, contours, -1, (0,255,0),1)

# cv.imshow("_", opening_img)
# cv.waitKey(0)

cv.destroyAllWindows()
