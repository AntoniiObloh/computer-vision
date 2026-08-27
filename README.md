# Computer Vision

An OpenCV learning project: notes in notebooks plus test data, models and scripts.

## Layout

```
opencv_start_course.ipynb   basics: reading images → HSV → threshold → histograms
document_scan_hdr_panorama.ipynb
                            document alignment, HDR, panoramas
tracking.ipynb              object tracking (BOOSTING/MIL/KCF/CSRT/TLD/MEDIANFLOW/GOTURN/MOSSE)
video_recording.ipynb       reading and writing video
deep_learning_based_object_detection.ipynb
                            object detection (SSD MobileNet v2, COCO)
human_pose_estimation.ipynb human pose (OpenPose BODY_25): heatmaps, PAFs, bottom-up and top-down

data/                       INPUT data (never overwritten by the code)
  images/                   all standalone test images
  hdr/                      a bracketed exposure series for HDR
  panorama/                 frames for panorama stitching
models/                     model weights (*.onnx, goturn.caffemodel + goturn.prototxt)
videos/                     input videos
outputs/                    RESULTS produced by the notebooks (overwritten)
HandTrackingProject/        MediaPipe Tasks hand tracking (module + runner)
PoseEstmationProject/       MediaPipe Tasks pose estimation (module + runner)
cv_logging.py               shared logging setup used by the projects above
scripts/                    standalone .py scripts
logs/                       run logs (camera_access, hand_tracking, pose_estimation)
.venv/                      virtual environment (not committed)
```

The notebooks sit in the project root on purpose — they are the entry points, and
every path inside them is relative to the root (`data/images/...`, `videos/...`,
`models/...`). That is why Jupyter has to be started **from the project root**.

## Setup from scratch

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name computer-vision \
    --display-name "Python 3.11 (Computer Vision venv)"
```

Check:

```bash
.venv/bin/python scripts/check_env.py
```

## Running

| What | How |
|---|---|
| Notebook | `.venv/bin/jupyter notebook` (see `how_to_run_jupyter.txt`) |
| Environment check | `.venv/bin/python scripts/check_env.py` |
| Camera with filters | `.venv/bin/python scripts/camera_access.py` |
| In PyCharm | Run → `check_env` or `Jupyter Notebook` |

## Environment

- Python 3.11.4 (arm64, python.org)
- OpenCV **4.10.0** (`opencv-contrib-python`) · NumPy 2.4.6 · matplotlib 3.10.7
- Jupyter Notebook 7.3.2

> ⚠️ It has to be `opencv-contrib-python`, not `opencv-python`: without contrib
> there is neither `cv2.legacy.*` (BOOSTING, TLD, MEDIANFLOW, MOSSE) nor GOTURN.
> Version 4.10 was chosen because 5.0 dropped the GOTURN tracker.

> ⚠️ The OpenCV 4.10 build ships without **AVIF** support, so
> `data/images/coco-cola.avif` and `data/images/building_windows.avif`
> currently cannot be read (`cv2.imread` returns `None`). They worked in 5.0.

> ℹ️ PyCharm **Community Edition** cannot open `.ipynb` as an interactive notebook —
> that is a Professional feature. So notebooks are edited in the browser through
> Jupyter, and PyCharm is used for `.py` files, autocompletion and debugging.

## Model weights

Large weights are not committed — they are downloaded by the first cell of the
corresponding notebook:

| Model | File | Size | Notebook |
|---|---|---|---|
| GOTURN | `models/goturn.caffemodel` | ~370 MB | `tracking.ipynb` |
| SSD MobileNet v2 | `models/ssd_mobilenet_v2_coco_2018_03_29/` | ~190 MB | `deep_learning_based_object_detection.ipynb` |
| OpenPose BODY_25 | `models/pose_body25/pose_iter_584000.caffemodel` | 104.7 MB | `human_pose_estimation.ipynb` |

BODY_25 is downloaded from a Hugging Face mirror — the official CMU host
(`posefs1.perception.cs.cmu.edu`) has not resolved for a long time. The connection
often breaks halfway, so `download()` in the notebook resumes from the point of
failure and verifies the final size: a truncated `.caffemodel` looks like a normal
file, but `cv2.dnn.readNetFromCaffe` fails on it.

## GOTURN

The weights (~370 MB) are not committed. The first cell of `tracking.ipynb`
downloads them into `models/` automatically from
[spmallick/goturn-files](https://github.com/spmallick/goturn-files) (the file is
split into 4 parts there). The old Dropbox link from the tutorials no longer
contains `goturn.caffemodel` — do not use it.
