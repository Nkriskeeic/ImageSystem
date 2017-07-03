import cv2
import numpy as np

class CharacterArea:
    def __init__(self):
        # 中心間のマハラノビス距離が30px以内ならOCR処理を行わない
        self.HASH_DISTANCE_ERROR = 30

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
            if len(approx) < 4 or approx.shape != (4,1,2):
                continue
            approxes.append(approx)
        return np.array(approxes, np.int32)

    def sort_area(self, area):
        points = np.reshape(area, (-1, 2))
        points = sorted(points, key=lambda x: x[1])
        top_points = sorted(points[:2], key=lambda x: x[0])
        bottom_points = sorted(points[2:4], key=lambda x: x[0])
        return np.array(top_points + bottom_points)

    def getBoundingBoxByPoints(self, sort_area):
        points = sort_area.reshape((-1, 2))
        left = min(points[0][0], points[2][0])
        right = max(points[1][0], points[3][0])
        top = min(points[0][1], points[1][1])
        bottom = max(points[2][1], points[3][1])
        return [(left, top), (right, bottom)]

    def getPartImageByRect(self, image, rect):
        return image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    def centers(self, contours):
        return contours. mean(axis=1)

    def getAreaInBoundingBox(self, area, bounding_box):
        return np.subtract(area, bounding_box[0])

    def setHash(self, contour_centers):
        return np.round(np.reshape(contour_centers, (-1, 2)), decimals=-1)

    def searchInHash(self, needle, hash_keys, answers):
        # hashが存在しなければFalse
        if len(hash_keys) == 0:
            return False
        # hash計算
        distances = np.abs(np.subtract(hash_keys, needle)).sum(axis=1)
        find_index = np.where(distances < self.HASH_DISTANCE_ERROR)
        if len(find_index) != 1 or len(answers) == 0 or len(answers) <= find_index[0][0]:
            return False
        else:
            return answers[find_index[0][0]]
