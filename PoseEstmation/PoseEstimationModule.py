import os
import cv2
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


model_path = 'pose_landmarker.task'
model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
if not os.path.isfile(model_path):
    print(f"downloading {model_path} ...")
    urllib.request.urlretrieve(model_url, model_path)
class poseDetector():
    def __init__(self, model_path=model_path):
        self.model_asset_path = model_path

        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.mpDraw = vision.drawing_utils

        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE)
        self.landmarker = self.PoseLandmarker.create_from_options(self.options)

    def findPose(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.imgRGB)
        self.landmarker_result = self.landmarker.detect(self.mp_img)

        if self.landmarker_result.pose_landmarks:
            for poseLms in self.landmarker_result.pose_landmarks:
                self.mpDraw.draw_landmarks(img, poseLms, vision.PoseLandmarksConnections.POSE_LANDMARKS)

        return img

    def findPosition(self, img, poseNo=0, pointId=None, draw=True):

        lmList = []
        if self.landmarker_result.pose_landmarks and poseNo < len(self.landmarker_result.pose_landmarks):
            poseLms = self.landmarker_result.pose_landmarks[poseNo]
            h, w, c = img.shape
            for id, lm in enumerate(poseLms):
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                lmList.append([id, center_x, center_y])

                if draw and (pointId is None or id == pointId):
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)

            return lmList

def main():
    cap = cv2.VideoCapture(
        '/Users/vadimoblog/Desktop/Computer Vision/videos/Megamind.mp4'
    )
    pTime = 0
    pose = poseDetector(model_path)
    while True:
        success, img = cap.read()
        if not success:
            break

        pose.findPose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()