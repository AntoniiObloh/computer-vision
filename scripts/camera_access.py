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

# Human-readable mode names - used only for legible log lines.
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
# so the log file always ends up in the same place no matter where the script
# is started from. parents[1] is the project root, since the script itself
# lives in scripts/. exist_ok=True: a repeated run does not fail if it exists.
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

HEARTBEAT = 100  # how often (in frames) to write a progress line

# ---------------------------------------------------------------------------
# Logging goes to two places at once: a file (a persistent session history)
# and stdout (to see the same lines in the terminal while it runs).
# The INFO level hides the log.debug calls below - deliberately, because those
# fire on every keypress. Switch to logging.DEBUG when debugging key handling.
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
log = logging.getLogger("camera")

# Corner detection parameters for FEATURES mode (Shi-Tomasi):
# maxCorners - the cap on how many points, qualityLevel - the corner "strength"
# threshold relative to the best one, minDistance - the minimum distance
# between points in pixels, blockSize - the window the corner is computed over.
feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 15, blockSize = 9)

# ---------------------------------------------------------------------------
# Video source from the command line. With no arguments - camera 0.
# ---------------------------------------------------------------------------
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    if s.isdigit():  # "1" -> camera index 1; anything else stays a file path / URL
        s = int(s)

# Separator in the log file: visually marks the start of a new session.
log.info("=" * 62)
log.info("session start | source=%r | opencv=%s", s, cv2.__version__)

# Loop state: the active filter and the "keep going" flag.
image_filter = PREVIEW
alive = True

# ---------------------------------------------------------------------------
# Opening the source. VideoCapture does not raise on failure, so the
# isOpened() check is mandatory - otherwise we get empty frames with no
# explanation.
# ---------------------------------------------------------------------------
source = cv2.VideoCapture(s)
if not source.isOpened():
    log.error("could not open source %r - exiting", s)
    sys.exit(1)

# The actual source parameters go to the log, so it is possible to tell later
# what the session was working with (for cameras the declared fps often does
# not match the real one).
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
declared_fps = source.get(cv2.CAP_PROP_FPS)
log.info(
    "opened: %dx%d @ %.1f fps | backend=%s",
    width, height, declared_fps, source.getBackendName(),
)

# The window is created up front: WINDOW_NORMAL makes it resizable.
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
log.info("keys: [p]review  [b]lur  [f]eatures  [c]anny  |  [q] or ESC to quit")

# Counters for the closing statistics. exit_reason is overwritten by whichever
# branch actually ends the loop; the initial value is the most common case.
frames = 0
started = time.monotonic()
exit_reason = "ESC pressed"

# ---------------------------------------------------------------------------
# Main loop. try/finally guarantees the camera is released and the windows are
# closed even on an exception or Ctrl+C - otherwise the device can stay busy
# until a reboot.
# ---------------------------------------------------------------------------
try:
    while alive:
        # Reading a frame. has_frame=False means end of file or a lost device -
        # both are a normal exit, not an exception.
        has_frame, frame = source.read()
        if not has_frame:
            exit_reason = "source returned no frame"
            log.warning("read() failed at frame %d - stream ended or device lost", frames + 1)
            break

        # Horizontal mirroring: with a webcam this feels natural, because the
        # picture behaves like a reflection rather than a view from the side.
        frame = cv2.flip(frame,1)

        # --- Frame processing for the active mode -------------------------
        result = frame
        if image_filter == PREVIEW:
            result = frame
        elif image_filter == CANNY:
            # Canny edge detector; 145/150 are the lower and upper thresholds.
            result = cv2.Canny(frame, 145,150)
        elif image_filter == BLUR:
            # Plain averaging over a 13x13 window.
            result = cv2.blur(frame, (13,13))
        elif image_filter == FEATURES:
            result = frame
            # Corner detection works on intensity, so convert to grayscale.
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
            if corners is not None:
                # reshape(-1,2) unfolds the (N,1,2) array into (x,y) pairs.
                # int() is required: OpenCV 5 rejects float center coordinates.
                for x,y in numpy.float32(corners).reshape(-1,2):
                    cv2.circle(result, (int(x), int(y)),10,(0,255,0),1)

        cv2.imshow(win_name, result)

        # --- Heartbeat: a periodic progress line --------------------------
        # Every HEARTBEAT frames, write the real average fps and the active
        # mode. That shows in the log that the session is alive, and how the
        # processing affects the speed.
        frames += 1
        if frames % HEARTBEAT == 0:
            elapsed = time.monotonic() - started
            log.info(
                "frame %d | filter=%s | %.1f fps avg",
                frames, FILTER_NAMES[image_filter], frames / elapsed if elapsed > 0 else 0.0,
            )

        # --- Keyboard -----------------------------------------------------
        # waitKey(1) is required not only for keys: without it the GUI never
        # repaints the window. The argument is a pause in milliseconds.
        key = cv2.waitKey(1)
        if key == -1:  # nothing pressed - the most common case, return immediately
            continue
        key &= 0xFF  # drop the high bits: some backends mix in modifiers
        if ord("A") <= key <= ord("Z"):  # Shift+key is handled as a plain key
            key += 32

        if key in (ESC, ord("q")):
            # Not break but alive=False: the loop ends normally, and the exit
            # reason is visible in the closing line in finally.
            exit_reason = "ESC pressed" if key == ESC else "'q' pressed"
            log.info("quit requested at frame %d (%s)", frames, exit_reason)
            alive = False
        elif key in KEY_FILTERS:
            new_filter = KEY_FILTERS[key]
            if new_filter == image_filter:
                # Pressing the same mode again goes to debug, so INFO does not
                # fill up with identical lines.
                log.debug("filter already %s - ignored", FILTER_NAMES[new_filter])
            else:
                log.info(
                    "filter %s -> %s at frame %d",
                    FILTER_NAMES[image_filter], FILTER_NAMES[new_filter], frames,
                )
                image_filter = new_filter
        else:
            # An unknown key breaks nothing, but leave a trace in debug.
            log.debug("unmapped key %d (%r) ignored", key, chr(key) if 32 <= key < 127 else "")
except KeyboardInterrupt:
    # Ctrl+C is an expected way to stop, hence a warning and not a traceback.
    exit_reason = "interrupted (Ctrl+C)"
    log.warning("interrupted by user at frame %d", frames)
except Exception:
    # log.exception writes the full traceback into the log file - without it
    # the cause of a crash cannot be recovered once the session is over.
    exit_reason = "unhandled exception"
    log.exception("aborted at frame %d", frames)
finally:
    # Cleanup runs on every exit path: release the device, close the windows
    # and write the session summary.
    elapsed = time.monotonic() - started
    source.release()
    cv2.destroyAllWindows()
    log.info(
        "session end | reason=%s | frames=%d | %.1f s | avg %.1f fps",
        exit_reason, frames, elapsed, frames / elapsed if elapsed > 0 else 0.0,
    )
