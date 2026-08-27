# ---------------------------------------------------------------------------
# Shared logging setup for the MediaPipe mini-projects (HandTrackingProject,
# PoseEstmationProject) and for anything else that runs a frame loop.
#
# The pattern is the same one scripts/camera_access.py established: every line
# goes to two places at once - a file under logs/ (a persistent session history)
# and stdout (so the same lines are visible in the terminal while the loop runs).
# It is pulled out into one module so both projects log identically and the
# format only has to be changed in a single place.
#
# The module only uses the standard library - no extra dependency in
# requirements.txt.
# ---------------------------------------------------------------------------
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The log directory is resolved from THIS file, not from the current working
# directory, so logs/ always ends up in the project root no matter where the
# script was started from (the projects are usually run from inside their own
# folder, PyCharm may use yet another cwd).
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"

# Default heartbeat: how often (in frames) a progress line is written.
# Every frame would flood the file; every 100 keeps it readable and still shows
# whether the loop is alive and how fast it goes.
HEARTBEAT = 100

LOG_FORMAT = "%(asctime)s  %(levelname)-8s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name, filename=None, level=None):
    """Return a logger writing to logs/<name>.log and to stdout.

    Calling it twice with the same name is safe: the second call finds the
    handlers already attached and returns the same logger. That is what lets a
    runner script and the module it imports share one log file - both simply
    ask for the same name.

    level=logging.DEBUG turns on the per-frame lines that are silent by
    default. It is applied on every call, not only the first one, so it works
    from either file no matter which of them happened to configure the logger.
    Left at None the level stays as it is (INFO for a fresh logger).
    """
    logger = logging.getLogger(name)

    # Handlers already attached - the logger has been configured by an earlier
    # call (e.g. by the imported detector module). Attaching them again would
    # duplicate every line in the file.
    if logger.handlers:
        if level is not None:
            logger.setLevel(level)
        return logger

    LOG_DIR.mkdir(exist_ok=True)  # a repeated run must not fail if it exists

    logger.setLevel(level if level is not None else logging.INFO)
    # propagate=False: the records stop here instead of also travelling up to
    # the root logger, which some libraries (and basicConfig) configure to print
    # on their own - without this the terminal shows every line twice.
    logger.propagate = False

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers = (
        logging.FileHandler(LOG_DIR / (filename or f"{name}.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _format_details(details):
    """Turn keyword arguments into the ` | key=value` tail of a log line."""
    return "".join(f" | {key}={value}" for key, value in details.items())


def log_session_start(logger, **details):
    """Write the session separator and header, and return the start timestamp.

    The returned value is meant to be kept and handed back to log_heartbeat /
    log_session_end:

        started_at = log_session_start(logger, source=0)

    time.monotonic() is used instead of time.time() because it cannot jump
    backwards when the system clock is adjusted - the measured duration stays
    correct.
    """
    logger.info("=" * 62)  # visually marks the start of a new session in the file
    logger.info("session start%s", _format_details(details))
    return time.monotonic()


def average_fps(frame_count, elapsed_seconds):
    """Frames per second over the whole session; 0.0 guards the first frame."""
    return frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0


def log_heartbeat(logger, frame_count, started_at, every=HEARTBEAT, **details):
    """Write a progress line every `every` frames, and nothing in between.

    The call sits inside the loop unconditionally - the "is it time yet" check
    lives here so the loop body stays readable.
    """
    if frame_count == 0 or frame_count % every != 0:
        return

    elapsed_seconds = time.monotonic() - started_at
    logger.info(
        "frame %d | %.1f fps avg%s",
        frame_count,
        average_fps(frame_count, elapsed_seconds),
        _format_details(details),
    )


def log_session_end(logger, started_at, frame_count, reason):
    """Write the closing summary: why it stopped, how long it ran, how fast.

    Belongs in a `finally` block, so the line is written on every exit path -
    a normal stop, Ctrl+C or a crash alike.
    """
    elapsed_seconds = time.monotonic() - started_at
    logger.info(
        "session end | reason=%s | frames=%d | %.1f s | avg %.1f fps",
        reason,
        frame_count,
        elapsed_seconds,
        average_fps(frame_count, elapsed_seconds),
    )
