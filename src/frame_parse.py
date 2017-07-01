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

    hash_list = np.array([])
    prev_answers = []

    # ここから各フレームを読み込み始める
    # 将来的にはここはwhile True:に置き換わる
    for _ in range(10):
        frame = cv2.imread(IMAGE_DIR + '/test.jpg', cv2.IMREAD_COLOR)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)
        blur_mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
        # green_frame = cv2.bitwise_and(frame, frame, mask=blur_mask)

        th_area = frame.shape[0] * frame.shape[1] / 100

        ca = CharacterArea()
        areas = ca.find(blur_mask, th_area)
        area_centers = ca.centers(areas)
        answers = []

        # 全てのareaについて文字列化を試みる
        for area, center in zip(areas, area_centers):
            # hashから探し出す
            answer = ca.searchInHash(center, hash_list, prev_answers)
            if answer:
                answers.append(answer)
            else:
                h, _ = cv2.findHomography(area, PTS_DST)
                warped_character_area = cv2.warpPerspective(frame, h, (CARD_SIZE[0], CARD_SIZE[1]))
                # TODO OCRで答えを得る．もし有効な答えを得られなければanswersにはFalseを代入する
                answers.append('apple')
        # この時点で全てのareaについてanswersに文字かFalseが代入されているので描画すればよい
        # TODO 描画処理

        prev_answers = answers
        hash_list = ca.setHash(area_centers)
