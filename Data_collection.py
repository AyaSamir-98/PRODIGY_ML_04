import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

folder = 'Data/please'
counter = 0
frame_rate = 0
offset = 30
imgsize = 300
while True:
    sucess, frame = cap.read()

    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue
        imgWhite = np.ones((imgsize,imgsize,3), np.uint8)*255

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize -wCal)/2)
            imgWhite[:,wGap:wGap+wCal] = imgResize
        else:
            k = imgsize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize -hCal)/2)
            imgWhite[hGap:hGap+hCal ,:] = imgResize

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
        """if frame_rate % 1 == 0:
            cv2.imwrite(f'{folder}/{frame_rate}.jpg', imgWhite)
        
        frame_rate += 1"""
    

    cv2.imshow('image',frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord('q'):
        break
