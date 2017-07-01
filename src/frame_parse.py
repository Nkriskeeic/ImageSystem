import cv2
from os import path as op
import numpy as np
from libs.Rectangle import Rectangle




if __name__ == '__main__':
    IMAGE_DIR = op.normpath(
        op.abspath(op.join(op.dirname(__file__), '../images/')))
    # apple = cv2.imread(IMAGE_DIR + '/apple.png', cv2.IMREAD_ANYCOLOR)
    # banana = cv2.imread(IMAGE_DIR + '/banana.png', cv2.IMREAD_ANYCOLOR)
    # chimpanzee = cv2.imread(IMAGE_DIR + '/chimpanzee.png', cv2.IMREAD_ANYCOLOR)

    frame = cv2.imread(IMAGE_DIR + '/test.jpg', cv2.IMREAD_COLOR)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    blur_mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
    green_frame = cv2.bitwise_and(frame, frame, mask=blur_mask)

    th_area = frame.shape[0] * frame.shape[1] / 100

    rectangle = Rectangle()
    areas = rectangle.find(blur_mask, th_area)
