import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
capture = cv2.VideoCapture(0)
hand_detector = htm.handDetector()
while True:
    success, image = capture.read()
    image = hand_detector.findHands(image)
    lmlist = hand_detector.findPosition(image, 0)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow(winname="Image", mat=image)
    cv2.waitKey(1)