import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import time

capture = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
mpDraw = vision.drawing_utils

pTime = 0
cTime = 0

while True:
    success, image = capture.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image =  mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)
    detection_result = detector.detect(mp_image)
    #print(detection_result.hand_landmarks)

    if detection_result.hand_landmarks:
        for handLms in detection_result.hand_landmarks:
            for id, lm in enumerate(handLms):
                print(id, lm.x, lm.y, lm.z)
            mpDraw.draw_landmarks(image, handLms, vision.HandLandmarksConnections.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(f"fps: {int(fps)}"), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
    cv2.imshow(winname="Image", mat = image)
    cv2.waitKey(1)