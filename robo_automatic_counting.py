import cv2
import numpy as np

# Load the image
img = cv2.imread('imgs/nuts.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to segment the image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)

# Apply a morphological operation to clean up the image
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours in the image
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a loop to count the objects based on shape and grey level
count = 0
for i, contour in enumerate(contours):
    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate the compactness of the contour
    compactness = 4*np.pi*area/(perimeter*perimeter)

    # Calculate the bounding box of the contour
    x,y,w,h = cv2.boundingRect(contour)

    # Extract the object from the original image
    obj = img[y:y+h, x:x+w]

    # Calculate the mean grey level of the object
    mean_gray = cv2.mean(obj)[0]

    # Check if the object meets the criteria for shape and grey level
    if compactness > 0.6 and area > 60 and mean_gray > 80:
        count += 1
        cv2.drawContours(img, contours, i, (0,255,0), 2)
        cv2.putText(img, str(count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Display the image with the counted objects
cv2.imshow('Objects counted', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
