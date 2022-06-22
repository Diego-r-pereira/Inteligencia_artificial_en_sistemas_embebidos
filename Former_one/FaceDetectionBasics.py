import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture("Video/1.mp4")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pTime = 0

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)