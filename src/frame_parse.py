import cv2
from os import path as op
import numpy as np
from libs.CharacterArea import CharacterArea

if __name__ == '__main__':
    # 定義
    IMAGE_DIR = op.normpath(
        op.abspath(op.join(op.dirname(__file__), '../images/')))
    # apple = cv2.imread(IMAGE_DIR + '/apple.png', cv2.IMREAD_ANYCOLOR)
    # banana = cv2.imread(IMAGE_DIR + '/banana.png', cv2.IMREAD_ANYCOLOR)
    # chimpanzee = cv2.imread(IMAGE_DIR + '/chimpanzee.png', cv2.IMREAD_ANYCOLOR)
    LOWER_GREEN = np.array([30, 30, 30])
    UPPER_GREEN = np.array([90, 255, 255])
    CARD_SIZE = [480, 360]
    PTS_DST = np.array([[[CARD_SIZE[0], 0]], [[0, 0]], [[0, CARD_SIZE[1]]], [[CARD_SIZE[0], CARD_SIZE[1]]]])

    frame = cv2.imread(IMAGE_DIR + '/test.jpg', cv2.IMREAD_COLOR)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)
    blur_mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
    green_frame = cv2.bitwise_and(frame, frame, mask=blur_mask)

    th_area = frame.shape[0] * frame.shape[1] / 100

    ca = CharacterArea()
    areas = ca.find(blur_mask, th_area)
    area_centers = ca.centers(areas)
    bounding_boxes = ca.getBoundingBoxByPoints(areas)
    for area, bounding_box in zip(areas, bounding_boxes):
        bb_image = ca.getPartImageByRect(frame, bounding_box)
        area_in_bb = ca.getAreaInBoundingBox(area, bounding_box)
        h, _ = cv2.findHomography(area_in_bb, PTS_DST)
        warped_character_area = cv2.warpPerspective(bb_image, h, (CARD_SIZE[0], CARD_SIZE[1]))