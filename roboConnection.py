import cv2
import time
import socket
import numpy as np
import math
from PIL import Image

server_cam_address = ('192.168.1.105', 11000)
#server_cam_address = ('localhost', 11000)

# Image size expected from camera
img_sizeX = 640
img_sizeY = 480
img_len = img_sizeX * img_sizeY * 3  # 211836
# print('Image size expected from camera: ', img_sizeX, 'x', img_sizeY, 'x 3 = ', img_len)

cont = 0
key = 0
while (key != 27):  # Press esc on Project window to finish

    try:
        # Camera server connection
        print("Connecting camera server...")
        sock_camera = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_camera.connect(server_cam_address)
        # print("Camera connected!")

        # Receiving image
        start = time.time()
        data = b''
        amount_received = len(data)
        while amount_received < img_len:
            new_data = sock_camera.recv(img_len - amount_received)
            if len(new_data) > 0:
                data += new_data
                amount_received += len(new_data)

        img_bytes = data[0:img_len]
        data = data[img_len:]
        img = Image.frombytes('RGB', (img_sizeX, img_sizeY), img_bytes)
        end = time.time()
        print('Image received in seconds: ', end - start)
        sock_camera.close()

        open_cv_image = np.array(img)
        img = open_cv_image[:, :, ::-1].copy()

        cv2.imshow("Press 's' to save this image as " + str(cont), img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('s'):
            cv2.imwrite('imgs/' + str(cont) + '.jpg', img)
            cont = cont + 1

    except Exception as e:
        # Clean up the connection
        print('Error: ' + str(e))
        sock_camera.close()
        exit(-1)

sock_camera.close()