import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import time

class handDetector():
    def __init__(self, model_asset_path = 'hand_landmarker.task', maxHands = 2):
        self.model_asset_path = model_asset_path
        self.maxHands = maxHands


        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                               num_hands=maxHands)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.mpDraw = vision.drawing_utils


    def findHands(self, image, draw=True):
        self.imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.imageRGB)
        self.detection_result = self.detector.detect(self.mp_image)

        if self.detection_result.hand_landmarks:
            for handLms in self.detection_result.hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, vision.HandLandmarksConnections.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNo = 0, pointId=None, draw = True):

        lmList = []
        if self.detection_result.hand_landmarks and handNo < len(self.detection_result.hand_landmarks):
            handLms = self.detection_result.hand_landmarks[handNo]
            h, w, c = image.shape
            for id, lm in enumerate(handLms):
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                lmList.append([id, center_x, center_y])

                if draw and (pointId is None or id == pointId):
                    cv2.circle(image, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

            return lmList


def main():
    pTime = 0
    cTime = 0
    capture = cv2.VideoCapture(0)
    hand_detector = handDetector()
    while True:
        success, image = capture.read()
        image = hand_detector.findHands(image)
        lmlist = hand_detector.findPosition(image, 0, 8)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow(winname="Image", mat=image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()