import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import traceback
import cv2
import numpy as np
import pyocr
from os import path as op
from PIL import Image
from libs.CharacterArea import CharacterArea

def distance(p1, p2):
    return np.abs(np.subtract(p1, p2)).sum()

if __name__ == '__main__':
    # 定義
    ESC_KEY = 27
    INTERVAL= 33
    FRAME_RATE = 30
    DEVICE_ID = 0
    WINDOW_NAME = "ImageSystem"
    IMAGE_DIR = op.normpath(
        op.abspath(op.join(op.dirname(__file__), '../images/')))

    img_dict = {}
    img_filenames = os.listdir("../images/")
    img_filenames.remove(".DS_Store")

    for filename in img_filenames:
        key  = filename.rstrip('.png')
        img  = cv2.imread(IMAGE_DIR+'/'+filename, -1)
        mask = cv2.cvtColor(img[:,:,3], cv2.COLOR_GRAY2BGR)
        mask = np.array(mask / 255.0, dtype=np.uint8)
        img  = img[:,:,:3]

        img_dict[key] = {'img' : img,
                         'w'   : img.shape[:2][0],
                         'h'   : img.shape[:2][1],
                         'mask': mask}

    LOWER_GREEN = np.array([30, 30, 30])
    UPPER_GREEN = np.array([90, 255, 255])
    CARD_WIDTH  = 480
    CARD_HEIGHT = 360
    PTS_DST_LITTLE = np.array([[0, 0], [CARD_WIDTH, 0], [0, CARD_HEIGHT], [CARD_WIDTH, CARD_HEIGHT]])
    PTS_DST_LEFT   = np.array([[0, CARD_HEIGHT], [0, 0],  [CARD_WIDTH, CARD_HEIGHT], [CARD_WIDTH, 0]])
    PTS_DST_RIGHT  = np.array([[CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, 0], [0, CARD_HEIGHT] ])

    hash_list = np.array([])
    prev_answers = []

    # Tesseractの用意
    tools = pyocr.get_available_tools()
    tool  = tools[0]

    # カメラの用意
    cap = cv2.VideoCapture(DEVICE_ID)
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape
    cv2.namedWindow(WINDOW_NAME)

    # 動画読み込み開始
    while end_flag == True:
        frame = c_frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)
        blur_mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
        # green_frame = cv2.bitwise_and(frame, frame, mask=blur_mask)

        th_area = frame.shape[0] * frame.shape[1] / 100
        ca = CharacterArea()

        try:
            areas = ca.find(blur_mask, th_area)
            if len(areas)<1:
                continue
            area_centers = ca.centers(areas)
            answers = []

            # 全てのareaについて文字列化
            for area, center in zip(areas, area_centers):
                # hashから探し出す
                answer = ca.searchInHash(center, hash_list, prev_answers)
                if answer:
                    answers.append(answer)
                else:
                    area = ca.sort_area(area)
                    if distance(area[0], area[1]) > distance(area[0], area[2]):
                        pts_dts = PTS_DST_LITTLE
                    elif area[0][1] < area[1][1]:
                        pts_dts = PTS_DST_RIGHT
                    else:
                        pts_dts = PTS_DST_LEFT

                    h, _ = cv2.findHomography(area, pts_dts)
                    warped_character_area = cv2.warpPerspective(frame, h, (CARD_WIDTH, CARD_HEIGHT))

                    # OCR処理
                    res = tool.image_to_string(Image.fromarray(warped_character_area),
                                    lang="en",
                                    builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))
                    if res and res[0].content in img_dict.keys():
                        answers.append(res[0].content)

            # オーバーレイ処理．frameを上書き
            print("Found:"+str(answers))
            for i, center in enumerate(area_centers):
                if len(answers)<=i:
                    break

                if answers[i] in img_dict.keys():
                    im = img_dict[answers[i]]
                    c_x = int(center[0][1])
                    c_y = int(center[0][0])

                    if c_frame[c_x-int(im['w']*0.5) : c_x-int(im['w']*0.5)+im['w'],
                               c_y-int(im['h']*0.5) : c_y-int(im['h']*0.5)+im['h']].shape == im['mask'].shape:

                        c_frame[c_x-int(im['w']*0.5) : c_x-int(im['w']*0.5)+im['w'],
                                c_y-int(im['h']*0.5) : c_y-int(im['h']*0.5)+im['h']] *= 1 - im['mask']

                        c_frame[c_x-int(im['w']*0.5) : c_x-int(im['w']*0.5)+im['w'],
                                c_y-int(im['h']*0.5) : c_y-int(im['h']*0.5)+im['h']] += im['img'] * im['mask']
        # どこかで死んだらエラー出力
        except Exception as e:
            ex, ms, tb = sys.exc_info()
            print(ms)
            traceback.print_tb(tb)

        cv2.imshow(WINDOW_NAME, c_frame)

        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        prev_answers = answers
        hash_list = ca.setHash(area_centers)
        end_flag, c_frame = cap.read()

    # カメラの解放
    cv2.destroyAllWindows()
    cap.release()
