import cv2
import mediapipe as mp

from mediapipe.tasks import python
from typing import Tuple, Union

from mediapipe.tasks.python import vision
import urllib.request
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
import sys
import os
import time
import math
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# The shared logging helpers live in the project root, one level above this
# folder. The projects are started from inside their own directory, so the root
# is not on sys.path by default and has to be added explicitly.
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_logging import get_logger, log_session_start, log_heartbeat, log_session_end

# One logger name for the whole project: the runner script (hand_tracking.py)
# asks for the same name, so both write into logs/hand_tracking.log.
logger = get_logger("face_detection")

model_path = 'blaze_face_short_range.tflite'
model_url  = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
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

class faceDetector():
    def __init__(self, model_path=model_path):
        self.MARGIN = 10  # pixels
        self.ROW_SIZE = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (255, 0, 0)  # red

        self.model_asset_path = model_path

        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetector = mp.tasks.vision.FaceDetector
        self.FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.mpDraw = vision.drawing_utils

        self.options = vision.FaceDetectorOptions(
            self.BaseOptions(model_asset_path=self.model_asset_path), running_mode=self.VisionRunningMode.IMAGE,
                    min_detection_confidence=0.5,
                    min_suppression_threshold=0.3,
                    result_callback=None)
        self.detector = vision.FaceDetector.create_from_options(self.options)

        # Last number of poses seen, so findPose reports only the changes
        # instead of writing a line for every frame.
        self.lastFaceCount = None

        logger.info("pose detector ready | model=%s | mode=IMAGE", Path(model_path).name)

    def _normalized_to_pixel_coordinates( self,
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def faceDetection(self, img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(mp_image)
        annotated_image = img.copy()
        height, width, _ = img.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
              keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                             width, height)
              color, thickness, radius = (0, 255, 0), 2, 2
              cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + bbox.origin_x,
                             self.MARGIN + self.ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)

        return annotated_image


def main():
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    # VideoCapture does not raise on a missing file - the check turns that into
    # one clear line instead of an empty window.
    if not cap.isOpened():
        logger.error("could not open source %r - exiting", video_source)
        sys.exit(1)

    pTime = 0
    started_at = log_session_start(
        logger, script="PoseEstimationModule.main", source=None, opencv=cv2.__version__,
    )
    face = faceDetector(model_path)

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

            annotated_image = face.faceDetection(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            frame_count += 1
            log_heartbeat(logger, frame_count, started_at, poses=face.lastFaceCount)

            cv2.putText(annotated_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", annotated_image)
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
