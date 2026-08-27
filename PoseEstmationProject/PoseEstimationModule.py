import os
import cv2
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The shared logging helpers live in the project root, one level above this
# folder. The projects are started from inside their own directory, so the root
# is not on sys.path by default and has to be added explicitly.
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_logging import get_logger, log_session_start, log_heartbeat, log_session_end

# One logger name for the whole project: the runner script (pose_estimation.py)
# asks for the same name, so both write into logs/pose_estimation.log.
logger = get_logger("pose_estimation")

model_path = 'pose_landmarker.task'
model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
if not os.path.isfile(model_path):
    # The path is relative, so the download lands in the current working
    # directory - the resolved path is logged to make that visible.
    logger.info("downloading %s -> %s", model_url, Path(model_path).resolve())
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception:
        # A failed download is the most common first-run problem; the traceback
        # in the log file is what makes it diagnosable afterwards.
        logger.exception("download of %s failed", model_path)
        raise
    logger.info("downloaded %s (%.1f MB)", model_path, os.path.getsize(model_path) / 1024 / 1024)
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

        # Last number of poses seen, so findPose reports only the changes
        # instead of writing a line for every frame.
        self.lastPoseCount = None

        logger.info("pose detector ready | model=%s | mode=IMAGE", Path(model_path).name)

    def findPose(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.imgRGB)
        self.landmarker_result = self.landmarker.detect(self.mp_img)

        # A pose entering or leaving the frame is a state change - worth an INFO
        # line. While it is simply being tracked, nothing is written.
        poseCount = len(self.landmarker_result.pose_landmarks) if self.landmarker_result.pose_landmarks else 0
        if poseCount != self.lastPoseCount:
            logger.info("poses detected: %d", poseCount)
            self.lastPoseCount = poseCount

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

        # Fires on every frame with nobody in view, so it is debug and not info -
        # visible only when get_logger is asked for logging.DEBUG.
        logger.debug("no landmarks for pose %d", poseNo)

def main():
    video_source = '/Users/vadimoblog/Desktop/Computer Vision/videos/Megamind.mp4'
    cap = cv2.VideoCapture(video_source)
    # VideoCapture does not raise on a missing file - the check turns that into
    # one clear line instead of an empty window.
    if not cap.isOpened():
        logger.error("could not open source %r - exiting", video_source)
        sys.exit(1)

    pTime = 0
    started_at = log_session_start(
        logger, script="PoseEstimationModule.main", source=Path(video_source).name, opencv=cv2.__version__,
    )
    pose = poseDetector(model_path)

    # Counters for the closing summary; exit_reason is overwritten by the branch
    # that actually ends the loop.
    frame_count = 0
    exit_reason = "loop ended"

    # try/finally guarantees the capture is released and the summary is written
    # even on an exception or Ctrl+C.
    try:
        while True:
            success, img = cap.read()
            if not success:
                # For a file this is simply the end of the video, not an error.
                exit_reason = "end of video"
                logger.info("no more frames after %d - end of source", frame_count)
                break

            pose.findPose(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            frame_count += 1
            log_heartbeat(logger, frame_count, started_at, poses=pose.lastPoseCount)

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        exit_reason = "interrupted (Ctrl+C)"
        logger.warning("interrupted by user at frame %d", frame_count)
    except Exception:
        exit_reason = "unhandled exception"
        logger.exception("aborted at frame %d", frame_count)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_session_end(logger, started_at, frame_count, exit_reason)

if __name__ == '__main__':
    main()
