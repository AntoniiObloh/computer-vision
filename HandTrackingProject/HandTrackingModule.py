import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
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

# One logger name for the whole project: the runner script (hand_tracking.py)
# asks for the same name, so both write into logs/hand_tracking.log.
logger = get_logger("hand_tracking")

class handDetector():
    def __init__(self, model_asset_path = 'hand_landmarker.task', maxHands = 2):
        self.model_asset_path = model_asset_path
        self.maxHands = maxHands

        # The model path is relative, so it only resolves when the script is
        # started from the project folder. Logging the absolute path turns the
        # usual "wrong cwd" failure into something readable in the log.
        resolved_model = Path(self.model_asset_path).resolve()
        if not resolved_model.is_file():
            logger.error("model file not found: %s", resolved_model)

        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                               num_hands=maxHands)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.mpDraw = vision.drawing_utils

        # Last number of hands seen, so findHands can report only the changes
        # instead of writing a line for every frame.
        self.lastHandCount = None

        logger.info("hand detector ready | model=%s | max_hands=%d", resolved_model.name, self.maxHands)


    def findHands(self, image, draw=True):
        self.imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.imageRGB)
        self.detection_result = self.detector.detect(self.mp_image)

        # Hands appearing or leaving the frame is a state change - worth an INFO
        # line. The unchanged case goes nowhere, so the log stays quiet while a
        # hand is simply being tracked.
        handCount = len(self.detection_result.hand_landmarks) if self.detection_result.hand_landmarks else 0
        if handCount != self.lastHandCount:
            logger.info("hands detected: %d", handCount)
            self.lastHandCount = handCount

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

        # Fires on every frame with no hand in view, so it is debug and not info -
        # visible only when get_logger is asked for logging.DEBUG.
        logger.debug("no landmarks for hand %d", handNo)


def main():
    pTime = 0
    cTime = 0
    capture = cv2.VideoCapture(0)
    # VideoCapture does not raise on failure, so the check is mandatory -
    # otherwise read() just returns empty frames with no explanation.
    if not capture.isOpened():
        logger.error("could not open camera 0 - exiting")
        sys.exit(1)

    started_at = log_session_start(logger, script="HandTrackingModule.main", source=0, opencv=cv2.__version__)
    hand_detector = handDetector()

    # Counters for the closing summary. exit_reason is overwritten by whichever
    # branch actually ends the loop.
    frame_count = 0
    exit_reason = "loop ended"

    # try/finally guarantees the camera is released and the summary is written
    # even on an exception or Ctrl+C - otherwise the device can stay busy.
    try:
        while True:
            success, image = capture.read()
            if not success:
                exit_reason = "source returned no frame"
                logger.warning("read() failed at frame %d - stream ended or device lost", frame_count + 1)
                break

            image = hand_detector.findHands(image)
            lmlist = hand_detector.findPosition(image, 0, 8)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            frame_count += 1
            log_heartbeat(logger, frame_count, started_at, hands=hand_detector.lastHandCount)

            cv2.putText(image, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow(winname="Image", mat=image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        # Ctrl+C is an expected way to stop, hence a warning and not a traceback.
        exit_reason = "interrupted (Ctrl+C)"
        logger.warning("interrupted by user at frame %d", frame_count)
    except Exception:
        # logger.exception writes the full traceback into the file - without it
        # the cause of a crash cannot be recovered once the session is over.
        exit_reason = "unhandled exception"
        logger.exception("aborted at frame %d", frame_count)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        log_session_end(logger, started_at, frame_count, exit_reason)

if __name__ == "__main__":
    main()
