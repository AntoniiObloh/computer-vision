# ---------------------------------------------------------------------------
# Dependencies: cv2 - image processing and the GUI window, numpy - point
# arrays, the rest is the standard library (arguments, time, logging, paths).
# ---------------------------------------------------------------------------
import cv2
import sys
import time
import logging
from pathlib import Path

import numpy

# ---------------------------------------------------------------------------
# Frame processing modes. Plain integers instead of an Enum - that is the
# original approach: the value is compared directly in the if/elif chain below.
# ---------------------------------------------------------------------------
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

# Human-readable mode names - used only for legible logger lines.
FILTER_NAMES = {PREVIEW: "preview", BLUR: "blur", FEATURES: "features", CANNY: "canny"}

# ---------------------------------------------------------------------------
# Key bindings. A table instead of an if/elif chain: adding a new mode is one
# line here, not an edit in three places inside the loop.
# ESC has no character literal, so it is given by its ASCII code.
# ---------------------------------------------------------------------------
ESC = 27
KEY_FILTERS = {
    ord("p"): PREVIEW,
    ord("b"): BLUR,
    ord("f"): FEATURES,
    ord("c"): CANNY,
}

# ---------------------------------------------------------------------------
# Log directory - in the project root, not in the current working directory,
# so the logger file always ends up in the same place no matter where the script
# is started_at from. parents[1] is the project root, since the script itself
# lives in scripts/. exist_ok=True: a repeated run does not fail if it exists.
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

HEARTBEAT = 100  # how often (in frame_count) to write a progress line

# ---------------------------------------------------------------------------
# Logging goes to two places at once: a file (a persistent session history)
# and stdout (to see the same lines in the terminal while it runs).
# The INFO level hides the logger.debug calls below - deliberately, because those
# fire on every keypress. Switch to logging.DEBUG when debugging pressed_key handling.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "camera_access.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("camera")

# Corner detection parameters for FEATURES mode (Shi-Tomasi):
# maxCorners - the cap on how many points, qualityLevel - the corner "strength"
# threshold relative to the best one, minDistance - the minimum distance
# between points in pixels, blockSize - the window the corner is computed over.
feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 15, blockSize = 9)

# ---------------------------------------------------------------------------
# Video source from the command line. With no arguments - camera 0.
# ---------------------------------------------------------------------------
video_source = 0
if len(sys.argv) > 1:
    video_source = sys.argv[1]
    if video_source.isdigit():  # "1" -> camera index 1; anything else stays a file path / URL
        video_source = int(video_source)

# Separator in the logger file: visually marks the start of a new session.
logger.info("=" * 62)
logger.info("session start | source=%r | opencv=%s", video_source, cv2.__version__)

# Loop state: the active filter and the "keep going" flag.
image_filter = PREVIEW
is_running = True

# ---------------------------------------------------------------------------
# Opening the source. VideoCapture does not raise on failure, so the
# isOpened() check is mandatory - otherwise we get empty frame_count with no
# explanation.
# ---------------------------------------------------------------------------
source = cv2.VideoCapture(video_source)
if not source.isOpened():
    logger.error("could not open source %r - exiting", video_source)
    sys.exit(1)

# The actual source parameters go to the logger, so it is possible to tell later
# what the session was working with (for cameras the declared fps often does
# not match the real one).
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
declared_fps = source.get(cv2.CAP_PROP_FPS)
logger.info(
    "opened: %dx%d @ %.1f fps | backend=%s",
    width, height, declared_fps, source.getBackendName(),
)

# The window is created up front: WINDOW_NORMAL makes it resizable.
window_name = "Camera Preview"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
logger.info("keys: [p]review  [b]lur  [f]eatures  [c]anny  |  [q] or ESC to quit")

# Counters for the closing statistics. exit_reason is overwritten by whichever
# branch actually ends the loop; the initial value is the most common case.
frame_count = 0
started_at = time.monotonic()
exit_reason = "ESC pressed"

# ---------------------------------------------------------------------------
# Main loop. try/finally guarantees the camera is released and the windows are
# closed even on an exception or Ctrl+C - otherwise the device can stay busy
# until a reboot.
# ---------------------------------------------------------------------------
try:
    while is_running:
        # Reading a frame. has_frame=False means end of file or a lost device -
        # both are a normal exit, not an exception.
        has_frame, frame = source.read()
        if not has_frame:
            exit_reason = "source returned no frame"
            logger.warning("read() failed at frame %d - stream ended or device lost", frame_count + 1)
            break

        # Horizontal mirroring: with a webcam this feels natural, because the
        # picture behaves like a reflection rather than a view from the side.
        frame = cv2.flip(frame,1)

        # --- Frame processing for the active mode -------------------------
        processed_frame = frame
        if image_filter == PREVIEW:
            processed_frame = frame
        elif image_filter == CANNY:
            # Canny edge detector; 145/150 are the lower and upper thresholds.
            processed_frame = cv2.Canny(frame, 145,150)
        elif image_filter == BLUR:
            # Plain averaging over a 13x13 window.
            processed_frame = cv2.blur(frame, (13,13))
        elif image_filter == FEATURES:
            processed_frame = frame
            # Corner detection works on intensity, so convert to grayscale.
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
            if corners is not None:
                # reshape(-1,2) unfolds the (N,1,2) array into (corner_x,corner_y) pairs.
                # int() is required: OpenCV 5 rejects float center coordinates.
                for corner_x,corner_y in numpy.float32(corners).reshape(-1,2):
                    cv2.circle(processed_frame, (int(corner_x), int(corner_y)),10,(0,255,0),1)

        cv2.imshow(window_name, processed_frame)

        # --- Heartbeat: a periodic progress line --------------------------
        # Every HEARTBEAT frame_count, write the real average fps and the active
        # mode. That shows in the logger that the session is is_running, and how the
        # processing affects the speed.
        frame_count += 1
        if frame_count % HEARTBEAT == 0:
            elapsed_seconds = time.monotonic() - started_at
            logger.info(
                "frame %d | filter=%s | %.1f fps avg",
                frame_count, FILTER_NAMES[image_filter], frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            )

        # --- Keyboard -----------------------------------------------------
        # waitKey(1) is required not only for keys: without it the GUI never
        # repaints the window. The argument is a pause in milliseconds.
        pressed_key = cv2.waitKey(1)
        if pressed_key == -1:  # nothing pressed - the most common case, return immediately
            continue
        pressed_key &= 0xFF  # drop the high bits: some backends mix in modifiers
        if ord("A") <= pressed_key <= ord("Z"):  # Shift+pressed_key is handled as a plain pressed_key
            pressed_key += 32

        if pressed_key in (ESC, ord("q")):
            # Not break but is_running=False: the loop ends normally, and the exit
            # reason is visible in the closing line in finally.
            exit_reason = "ESC pressed" if pressed_key == ESC else "'q' pressed"
            logger.info("quit requested at frame %d (%s)", frame_count, exit_reason)
            is_running = False
        elif pressed_key in KEY_FILTERS:
            new_filter = KEY_FILTERS[pressed_key]
            if new_filter == image_filter:
                # Pressing the same mode again goes to debug, so INFO does not
                # fill up with identical lines.
                logger.debug("filter already %s - ignored", FILTER_NAMES[new_filter])
            else:
                logger.info(
                    "filter %s -> %s at frame %d",
                    FILTER_NAMES[image_filter], FILTER_NAMES[new_filter], frame_count,
                )
                image_filter = new_filter
        else:
            # An unknown pressed_key breaks nothing, but leave a trace in debug.
            logger.debug("unmapped key %d (%r) ignored", pressed_key, chr(pressed_key) if 32 <= pressed_key < 127 else "")
except KeyboardInterrupt:
    # Ctrl+C is an expected way to stop, hence a warning and not a traceback.
    exit_reason = "interrupted (Ctrl+C)"
    logger.warning("interrupted by user at frame %d", frame_count)
except Exception:
    # logger.exception writes the full traceback into the logger file - without it
    # the cause of a crash cannot be recovered once the session is over.
    exit_reason = "unhandled exception"
    logger.exception("aborted at frame %d", frame_count)
finally:
    # Cleanup runs on every exit path: release the device, close the windows
    # and write the session summary.
    elapsed_seconds = time.monotonic() - started_at
    source.release()
    cv2.destroyAllWindows()
    logger.info(
        "session end | reason=%s | frames=%d | %.1f s | avg %.1f fps",
        exit_reason, frame_count, elapsed_seconds, frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
    )
