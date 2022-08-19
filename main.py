import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# for webcam
cap = cv2.VideoCapture(0)
# size
cap.set(3, 1280)  #width
cap.set(4, 720)   #height

# Hand Detector
# 80% sure then detect the hand
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find Function - that can relate x and y
# x is the raw distance y is the value in cm
# measuring tape

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# quadratic equation for the graph
# the relationship between x and y
# whatever value we will get in x can be used to find the value of y in cm

coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# Loop
while True:
    success, img = cap.read()
    # only display text
    hands = detector.findHands(img, draw=False)

    if hands:
        # only one hand
        lmList = hands[0]['lmList']  # land mark list- list of all the points
        # print(lmlist)

        x, y, w, h = hands[0]['bbox']  # bounding box

        # index numbers for the hand measurements
        x1, y1 = lmList[5]       # the points
        x2, y2 = lmList[17]

        # print(abs(x2-x1))
        # for diagonal measurement
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        # print(distanceCM, distance)

        # bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)

        # will put text on the image in a string
        # an offset
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))


    # display
    cv2.imshow("Image", img)

    # 1 ms delay
    cv2.waitKey(1)