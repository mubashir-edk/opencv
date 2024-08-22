import cv2 as cv
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1080, 920

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    image = cv.resize(image, (200,275))
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
# print(len(overlayList))

pTime = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    
    if len(lmList) != 0:
        fingers = []
        
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
    
        h, w, c = overlayList[totalFingers].shape
        img[0:h,0:w] = overlayList[totalFingers]
        
        cv.putText(img, str(totalFingers), (50,550), cv.FONT_HERSHEY_PLAIN, 5, (0,255,0), 3)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv.putText(img, f'fps: {int(fps)}', (25,350), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    
    cv.imshow('Video', img)
    cv.waitKey(1)