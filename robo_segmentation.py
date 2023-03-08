import cv2

# Load the image
img = cv2.imread('robo_table.png')

# Define the window name and create a window
manual_selection = 'Image segmentation'
cv2.namedWindow(manual_selection)

# Define the default grey levels for the background and object
bg_level = 100
obj_level = 200

# Create a function to handle trackbar events
def on_trackbar(val):
    pass


# Create trackbars for the background and object levels
cv2.createTrackbar('Background level', manual_selection, bg_level, 255, on_trackbar)
cv2.createTrackbar('Object level', manual_selection, obj_level, 255, on_trackbar)

# Create a loop to update the image based on the trackbar values
while True:
    # Get the current trackbar values
    bg_level = cv2.getTrackbarPos('Background level', manual_selection)
    obj_level = cv2.getTrackbarPos('Object level', manual_selection)

    # Create a binary mask based on the trackbar values
    mask = cv2.inRange(img, bg_level, obj_level)

    # Show the masked image
    cv2.imshow(manual_selection, mask)

    # Wait for a key press and check if 'q' was pressed to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up and exit 
cv2.destroyAllWindows()