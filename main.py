import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
# from tkinter import *
# from PIL import Image, ImageTk

image = cv.imread("imgs/robo_table.png")
assert image is not None, "file could not be read, check with os.path.exists()"

cropped_image = image[220:430, 340:430] # [220:420, 340:430] for conveyor belt 

# Save the cropped image
cv.imwrite("imgs/cropped_robo.png", cropped_image)
gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

# blur_image = cv.medianBlur(gray_image, 3)

ret, thresh_img = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = kernel = np.ones((3,3),np.uint8)

# Opening is just another name of erosion followed by dilation
opening_img = cv.morphologyEx(thresh_img,cv.MORPH_OPEN,kernel, iterations = 2)

edges = cv.Canny(opening_img, 30, 200)

contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(cropped_image, contours, -1, (0,255,0),1)

# cv.imshow("_", opening_img)
# cv.waitKey(0)

for i, contour in enumerate(contours):
    
    if i == 0:
        continue # skip the first contour since it stores the whole image..

    # Aprroximate the shapes
    # epsilon specifies the precision that the approx-function will use. True tells the function that it is supposed to be a closed shape
    epsilon = 0.01 * cv.arcLength(contour, True) 
    # Uses Douglas-Peucker algorithm to approx the shapes with the epsilon precision and the same True effect
    approx = cv.approxPolyDP(contour, epsilon, True) 

    # find the coordinates of each ahspe and make som approximate
    # calculations to place the text ish in the middle
    x, y, width, height = cv.boundingRect(approx)
    x_mid = int(x + (width / 3))
    y_mid = int(y + (height / 1.5))

    # store coordinates
    coordinates = (x_mid, y_mid)
    # choose color and font
    color = (0, 255, 0)
    font = cv.FONT_HERSHEY_DUPLEX

    area = cv.contourArea(contour)
    radius = np.sqrt(area / np.pi)

    perimeter = cv.arcLength(contour,True)

    pi_relationship = perimeter / (2 * radius)

    print(pi_relationship)
    print()
    
    if math.isclose(pi_relationship, np.pi, abs_tol = 0.2):
        cv.putText(cropped_image, "C", coordinates, font, 0.3, color, 1)
    else:
        cv.putText(cropped_image, "S", coordinates, font, 0.3, color, 1)

cv.imshow("_", cropped_image)

cv.waitKey(0)
cv.destroyAllWindows()