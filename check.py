import pandas as pd
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


# class_names = os.listdir('Data')

# df = pd.DataFrame(class_names, columns=['label_name'])

# print(df)

# df.to_csv('labels.csv')

# with open('labels.csv', 'r') as f_in, open('label.txt', 'w') as f_out:
#     content = f_in.read().replace(',', ' ')
#     f_out.write(content)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("hand.h5","label.txt")
counter = 0
labels = os.listdir('Data')
offset = 20
imgsize = 300
while True:
    sucess, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3), np.uint8)*255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue  

        imgCropShape = imgCrop.shape

        ascpectRatio = h/w

        if ascpectRatio > 1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            imgWhite[:,wGap:wGap+wCal] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)
        else:
            k = imgsize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize-hCal)/2)
            imgWhite[hGap:hGap+hCal, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+150,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

    cv2.imshow("ImageCrop",imgOutput)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break