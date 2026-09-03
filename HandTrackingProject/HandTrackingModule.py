"""Hand tracking with MediaPipe Tasks, wrapped in a reusable detector class.

Fully annotated: every function has parameter and return types, and the
detector's state is declared on the class instead of appearing mid-method.

    Run:    .venv/bin/python HandTrackingModule.py     (from this folder)
    Check:  mypy HandTrackingProject/
"""

import sys
import time
from pathlib import Path
from typing import Final

import cv2
from cv2.typing import MatLike

# mediapipe ships no py.typed marker, so a checker cannot see inside it and
# reports "missing library stubs". The ignore is per-import and narrowed to
# that one error code: any OTHER problem on these lines is still reported.
# Everything coming out of these modules is typed Any, which is why the wrapper
# below annotates its own boundaries — that is where the types come back.
import mediapipe as mp  # type: ignore[import-untyped]
from mediapipe.tasks import python  # type: ignore[import-untyped]
from mediapipe.tasks.python import vision  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# The shared logging helpers live in the project root, one level above this
# folder. The projects are started from inside their own directory, so the root
# is not on sys.path by default and has to be added explicitly.
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_logging import (  # noqa: E402  (import must follow the sys.path line)
    get_logger,
    log_heartbeat,
    log_session_end,
    log_session_start,
)

# One logger name for the whole project: the runner script (hand_tracking.py)
# asks for the same name, so both write into logs/hand_tracking.log.
logger = get_logger("hand_tracking")

# Final marks a constant: a checker rejects any later assignment to these
# names, which is the difference between a constant and a module-level variable.
DEFAULT_MODEL_PATH: Final[str] = "hand_landmarker.task"
DEFAULT_MAX_HANDS: Final[int] = 2
LANDMARK_COLOR: Final[tuple[int, int, int]] = (255, 0, 255)

# A named alias for the awkward return shape of findPosition: one entry per
# landmark, each [landmark_id, x_px, y_px]. The alias means the type is written
# once and read by name everywhere else.
# NOTE: a NamedTuple would describe this far better than three anonymous ints,
# since it would name the fields and stop `point[1]` from meaning anything.
# That changes how callers unpack the result, so it is left as a next step.
type LandmarkList = list[list[int]]


class handDetector:
    """Wraps a MediaPipe HandLandmarker and the drawing on top of it."""

    # State shared between methods, declared up front. `| None` is not
    # decoration: it records that both are empty until findHands has run once,
    # which is exactly the precondition findPosition depends on.
    detection_result: vision.HandLandmarkerResult | None
    lastHandCount: int | None

    def __init__(
        self,
        model_asset_path: str | Path = DEFAULT_MODEL_PATH,
        maxHands: int = DEFAULT_MAX_HANDS,
    ) -> None:
        # Accept either a str or a Path — callers should not have to convert —
        # but store the one concrete type the rest of the class relies on.
        self.model_asset_path: str = str(model_asset_path)
        self.maxHands: int = maxHands

        # The model path is relative, so it only resolves when the script is
        # started from the project folder. Logging the absolute path turns the
        # usual "wrong cwd" failure into something readable in the log.
        resolved_model = Path(self.model_asset_path).resolve()
        if not resolved_model.is_file():
            logger.error("model file not found: %s", resolved_model)

        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            num_hands=self.maxHands,
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.mpDraw = vision.drawing_utils

        self.detection_result = None
        # Last number of hands seen, so findHands can report only the changes
        # instead of writing a line for every frame.
        self.lastHandCount = None

        logger.info(
            "hand detector ready | model=%s | max_hands=%d",
            resolved_model.name,
            self.maxHands,
        )

    def findHands(self, image: MatLike, draw: bool = True) -> MatLike:
        """Run detection on one BGR frame and return it, optionally drawn on."""
        # Per-frame scratch values, so plain locals rather than attributes:
        # nothing outside this method reads them, and keeping them off `self`
        # means the class has two pieces of state to reason about instead of
        # five. FaceDetectionModule already does it this way.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)
        self.detection_result = self.detector.detect(mp_image)

        # Hands appearing or leaving the frame is a state change - worth an INFO
        # line. The unchanged case goes nowhere, so the log stays quiet while a
        # hand is simply being tracked.
        handCount = (
            len(self.detection_result.hand_landmarks)
            if self.detection_result.hand_landmarks
            else 0
        )
        if handCount != self.lastHandCount:
            logger.info("hands detected: %d", handCount)
            self.lastHandCount = handCount

        if self.detection_result.hand_landmarks:
            for handLms in self.detection_result.hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image,
                        handLms,
                        vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    )

        return image

    def findPosition(
        self,
        image: MatLike,
        handNo: int = 0,
        pointId: int | None = None,
        draw: bool = True,
    ) -> LandmarkList | None:
        """Pixel coordinates of one hand's landmarks, or None if there is none.

        The `| None` in the return type is not a style choice — the function
        really does fall off the end without a `return` when no hand is in
        view, and Python turns that into None. Spelling it out forces callers
        to handle the empty case instead of hitting it at runtime.
        """
        # findHands has not run yet, so there is nothing to read. Without the
        # declared `detection_result: ... | None` this was an AttributeError
        # waiting for the first caller who got the call order wrong.
        if self.detection_result is None:
            logger.debug("findPosition called before findHands")
            return None

        lmList: LandmarkList = []
        if self.detection_result.hand_landmarks and handNo < len(
            self.detection_result.hand_landmarks
        ):
            handLms = self.detection_result.hand_landmarks[handNo]
            h, w, c = image.shape
            for id, lm in enumerate(handLms):
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                lmList.append([id, center_x, center_y])

                if draw and (pointId is None or id == pointId):
                    cv2.circle(image, (center_x, center_y), 15, LANDMARK_COLOR, cv2.FILLED)

            return lmList

        # Fires on every frame with no hand in view, so it is debug and not info -
        # visible only when get_logger is asked for logging.DEBUG.
        logger.debug("no landmarks for hand %d", handNo)
        return None


def main() -> None:
    # `-> None` is what puts this body under the checker at all: by default
    # mypy skips the bodies of unannotated functions, so the float/int mix
    # below went unreported here while the identical line in hand_tracking.py
    # (module level, always checked) was flagged.
    #
    # Annotated as float, not left to be inferred as int from `0`: the very
    # next assignment is time.time(), which is a float.
    pTime: float = 0.0
    cTime: float = 0.0
    capture = cv2.VideoCapture(0)
    # VideoCapture does not raise on failure, so the check is mandatory -
    # otherwise read() just returns empty frames with no explanation.
    if not capture.isOpened():
        logger.error("could not open camera 0 - exiting")
        sys.exit(1)

    started_at = log_session_start(
        logger, script="HandTrackingModule.main", source=0, opencv=cv2.__version__
    )
    hand_detector = handDetector()

    # Counters for the closing summary. exit_reason is overwritten by whichever
    # branch actually ends the loop.
    frame_count: int = 0
    exit_reason: str = "loop ended"

    # try/finally guarantees the camera is released and the summary is written
    # even on an exception or Ctrl+C - otherwise the device can stay busy.
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
            # Called for the drawing side effect - it puts the marker on
            # landmark 8, the index fingertip. The return value is unused here.
            hand_detector.findPosition(image, 0, 8)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            frame_count += 1
            log_heartbeat(
                logger, frame_count, started_at, hands=hand_detector.lastHandCount
            )

            cv2.putText(
                image,
                f"fps: {int(fps)}",
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                LANDMARK_COLOR,
                3,
            )
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
