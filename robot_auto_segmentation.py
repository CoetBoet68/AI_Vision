import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math as math
import imutils

image = cv.imread("imgs/robo_table.png")
assert image is not None, "file could not be read, check with os.path.exists()"

# Crop picture to focus on the objects
cropped_image = image[220:430, 340:430] # [220:420, 340:430] for conveyor belt 

# Save the cropped image
cv.imwrite("imgs/cropped_robot.png", cropped_image)

# Create grayscale image
gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

# Gaussian blur
blur = cv.GaussianBlur(cropped_image,(5,5),0)

# Otzu thersholding after gaussian blur
ret, th_img = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)

# Opening is just another name of erosion followed by dilation
opening_img = cv.morphologyEx(th_img, cv.MORPH_OPEN, kernel, iterations = 2)

images = [
    cropped_image, 0, gray_image,
    blur, 0, th_img,
    blur, 0, opening_img
    ]

titles = [
    "Original noisy cropped image", "Histogram", "Gray image",
    "Gausian blur", "Histogram", "Otzu Thresholding",
    "Gausian blur", "Histogram", "Segmentation using opening"
]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

# choose color and font for visualizations
color = (0, 255, 0)
font = cv.FONT_HERSHEY_DUPLEX

def check_if_bad(contour):
    area = cv.contourArea(contour)
    if area < 100:
        return True

def check_if_circle(contour):

     # Calculate area of the contour using Greens theorem 
    area = cv.contourArea(contour)
    areas.append(area)

    # Calulate the radius using the area => r = (A / pi)^(1/2)
    radius = np.sqrt(area / np.pi)

    # Calculate the perimeter of a closed contour perimeter (True states that the contour is closed)
    perimeter = cv.arcLength(contour, True)

    # Calculate the relationship between perimeter and diameter (pi = C / D)
    pi_rel = perimeter / (2 * radius)

     # Sort out possible "open contours" by setting lower limit as 100 
    if area > 100:
        if math.isclose(pi_rel, np.pi, abs_tol = 0.21):
            return True
        else:
            return False
    else:
        return None

areas = []
shapes = []

edges = cv.Canny(opening_img, 30, 200)

# Find the contours in the binary picture, RETR_EXTERNAL retrieves only the extreme outer contours, CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_copy = cropped_image.copy()

mask = np.ones(img_copy.shape[:2], dtype = "uint8") * 255

cv.drawContours(img_copy, contours, -1, (0,255,0),1)

# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(img_copy)
plt.axis('off')
plt.title("Before masking")

for contour in contours:
    if check_if_bad(contour):
        cv.drawContours(mask, [contour], -1, 0, -1)

img_copy = cv.bitwise_and(img_copy, img_copy, mask=mask)
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(mask)
plt.axis('off')
plt.title("Mask")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(img_copy)
plt.axis('off')
plt.title("After masking")

for i, contour in enumerate(contours):
    
    # Skip the first contour as it stores the whole image
    if i == 0:
        continue
    
    # Aprroximate the shapes
    # epsilon specifies the precision that the approx-function will use. True tells the function that it is supposed to be a closed shape
    epsilon = 0.05 * cv.arcLength(contour, True)

    # Uses Douglas-Peucker algorithm to approx the shapes with the epsilon precision and the same True effect
    approx = cv.approxPolyDP(contour, epsilon, True)

    # find the coordinates of each shape and make some approximate
    # calculations to place the text ish in the middle of the object
    x, y, width, height = cv.boundingRect(approx)
    x_mid = int(x + (width / 3))
    y_mid = int(y + (height / 1.5))

    # Store the approximative coordinates in a tuple
    coordinates = (x_mid, y_mid)

    # Sort out possible "open contours" by setting lower limit as 100 
    if check_if_circle(contour): 
        cv.putText(img_copy, f"C", coordinates, font, 0.3, color, 1)
    elif check_if_circle(contour) is False:
        cv.putText(img_copy, f"S", coordinates, font, 0.3, color, 1)

    # cv.imshow(f"Object number {i} before masking", img_copy)
    # cv.waitKey(0)
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(img_copy)
plt.axis('off')
plt.title("Classification")

plt.show()

cv.destroyAllWindows()