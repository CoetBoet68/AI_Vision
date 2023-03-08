import cv2 # OpenCV module
ip_camara_dev = 'http://VxC:vxc2018@158.42.206.2/mjpeg/snap.cgi?chn=0'

# Open device
print("Starting acquisition")
video = cv2.VideoCapture(ip_camara_dev)

cont = 0
key = 0
not_working = False
while (key != 27 and not_working == False):  # Press esc on the image window to finish

    try:
        # Image acquisition: a BGR image is captured
        ret_ok, img = video.read()

        if ret_ok:

            cv2.imshow("Press 's' to save this image as " + str(cont), img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('s'):
                cv2.imwrite('imgs/' + str(cont) + '.jpg', img)
                cont = cont + 1

        else:
            print('Image acquisition error...')
            not_working = True

    except Exception as e:
        # Clean up the connection
        print('Error: ' + str(e))
        video.release()
        exit(-1)

# Close device
video.release()