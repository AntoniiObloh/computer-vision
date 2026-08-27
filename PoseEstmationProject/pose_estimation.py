import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import sys
import time
from pathlib import Path

# The shared logging helpers live in the project root, one level up - the script
# is started from inside this folder, so the root is not on sys.path by default.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_logging import get_logger, log_session_start, log_heartbeat, log_session_end

import PoseEstimationModule as rsm

# Same logger name as in PoseEstimationModule: the handlers are already attached
# by the import above, so this returns the very same logger and both write into
# logs/pose_estimation.log.
logger = get_logger("pose_estimation")

pTime = 0
cTime = 0
capture = cv2.VideoCapture(0)
# VideoCapture does not raise on failure - without this check the loop would
# just spin on empty frames.
if not capture.isOpened():
    logger.error("could not open camera 0 - exiting")
    sys.exit(1)

started_at = log_session_start(logger, script="pose_estimation.py", source=0, opencv=cv2.__version__)
hand_detector = rsm.poseDetector()

# Counters for the closing summary; exit_reason is overwritten by the branch
# that actually ends the loop.
frame_count = 0
exit_reason = "loop ended"

# try/finally: the camera is released and the summary written on every exit
# path - a normal stop, Ctrl+C or a crash alike.
try:
    while True:
        success, image = capture.read()
        if not success:
            exit_reason = "source returned no frame"
            logger.warning("read() failed at frame %d - stream ended or device lost", frame_count + 1)
            break

        image = hand_detector.findPose(image)
        lmlist = hand_detector.findPosition(image, 0)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        frame_count += 1
        # One progress line every 100 frames: real average fps and how many
        # poses are currently in view.
        log_heartbeat(logger, frame_count, started_at, poses=hand_detector.lastPoseCount)

        cv2.putText(image, str(f"fps: {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow(winname="Image", mat=image)
        cv2.waitKey(1)
except KeyboardInterrupt:
    # Ctrl+C is the normal way to stop this script, so it is a warning and not
    # a traceback.
    exit_reason = "interrupted (Ctrl+C)"
    logger.warning("interrupted by user at frame %d", frame_count)
except Exception:
    # The full traceback goes into the log file - otherwise the cause of a crash
    # is lost once the terminal is closed.
    exit_reason = "unhandled exception"
    logger.exception("aborted at frame %d", frame_count)
finally:
    capture.release()
    cv2.destroyAllWindows()
    log_session_end(logger, started_at, frame_count, exit_reason)
