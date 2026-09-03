"""Runner script: opens camera 0 and draws hand landmarks until stopped.

The detection itself lives in HandTrackingModule — this file is only the loop,
the FPS counter and the session logging around them.

    Run:    .venv/bin/python hand_tracking.py     (from this folder)
    Check:  mypy HandTrackingProject/
"""

import sys
import time
from pathlib import Path
from typing import Final

import cv2

# The shared logging helpers live in the project root, one level up - the script
# is started from inside this folder, so the root is not on sys.path by default.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_logging import (  # noqa: E402  (import must follow the sys.path line)
    get_logger,
    log_heartbeat,
    log_session_end,
    log_session_start,
)

import HandTrackingModule as htm  # noqa: E402  (same reason)

# Same logger name as in HandTrackingModule: the handlers are already attached
# by the import above, so this returns the very same logger and both write into
# logs/hand_tracking.log.
logger = get_logger("hand_tracking")

CAMERA_INDEX: Final[int] = 0
TEXT_COLOR: Final[tuple[int, int, int]] = (255, 0, 255)

# Annotated as float rather than inferred as int from `0`. This is the one
# error mypy already reported on the previous version of this file:
#
#   hand_tracking.py:52: error: Incompatible types in assignment
#   (expression has type "float", variable has type "int")  [assignment]
#
# `pTime = 0` made mypy infer int, and `pTime = cTime` then assigned a float
# into it. Harmless at runtime, but it is the same class of mistake that does
# bite when the two types are not silently compatible.
pTime: float = 0.0
cTime: float = 0.0

capture = cv2.VideoCapture(CAMERA_INDEX)
# VideoCapture does not raise on failure - without this check the loop would
# just spin on empty frames.
if not capture.isOpened():
    logger.error("could not open camera %d - exiting", CAMERA_INDEX)
    sys.exit(1)

started_at = log_session_start(
    logger, script="hand_tracking.py", source=CAMERA_INDEX, opencv=cv2.__version__
)
hand_detector = htm.handDetector()

# Counters for the closing summary; exit_reason is overwritten by the branch
# that actually ends the loop.
frame_count: int = 0
exit_reason: str = "loop ended"

# try/finally: the camera is released and the summary written on every exit
# path - a normal stop, Ctrl+C or a crash alike.
try:
    while True:
        success, image = capture.read()
        if not success:
            exit_reason = "source returned no frame"
            logger.warning(
                "read() failed at frame %d - stream ended or device lost",
                frame_count + 1,
            )
            break

        image = hand_detector.findHands(image)
        # Called for the drawing side effect; the returned coordinates are not
        # used here. When they are needed the declared return type is
        # `htm.LandmarkList | None`, so the None case has to be handled:
        #     lmlist: htm.LandmarkList | None = hand_detector.findPosition(image, 0)
        #     if lmlist is not None:
        #         ...
        hand_detector.findPosition(image, 0)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        frame_count += 1
        # One progress line every 100 frames: real average fps and how many
        # hands are currently in view.
        log_heartbeat(logger, frame_count, started_at, hands=hand_detector.lastHandCount)

        cv2.putText(
            image,
            f"fps: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            TEXT_COLOR,
            3,
        )
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
