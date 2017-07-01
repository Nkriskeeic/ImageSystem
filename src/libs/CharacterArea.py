import cv2
import numpy as np

class CharacterArea:
    def _findObject(self, mask, th_area):
        _, contours, _ = cv2.findContours(mask,
                                                      cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        return list(filter(lambda c: cv2.contourArea(c) > th_area, contours))

    def find(self, mask, th_area):
        # 長方形の座標を返す
        approxes = []

        contours_large = self._findObject(mask, th_area)

        for (i, cnt) in enumerate(contours_large):
            arclen = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True)
            if len(approx) < 4:
                continue
            approxes.append(approx)
        return np.array(approxes, np.int32)

    def getBoundingBoxByPoints(self, contours):
        bounding_boxes = []
        for contour in contours:
            points = list(map(lambda x: x[0], contour))
            points = sorted(points, key=lambda x: x[1])
            top_points = sorted(points[:2], key=lambda x: x[0])
            bottom_points = sorted(points[2:4], key=lambda x: x[0])
            points = top_points + bottom_points

            left = min(points[0][0], points[2][0])
            right = max(points[1][0], points[3][0])
            top = min(points[0][1], points[1][1])
            bottom = max(points[2][1], points[3][1])
            bounding_boxes.append([(left, top), (right, bottom)])
        return bounding_boxes

    def getPartImageByRect(self, image, rect):
        return image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    def centers(self, contours):
        return contours. mean(axis=1)

    def getAreaInBoundingBox(self, area, bounding_box):
        return np.subtract(area, bounding_box[0])