from IPython.display import Image, display, clear_output
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

img = cv2.imread('imgs/nuts.png')
print('Image Shape:', img.shape)
print('Image type', type(img), ' of ', img.dtype)
