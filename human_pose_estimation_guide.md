# Human Pose Estimation: OpenPose BODY_25 with `cv2.dnn`

A guide for the `human_pose_estimation.ipynb` notebook. The idea: you write the
code yourself, and this document explains what each block has to do, why it is
done that way, where the traps are, and which numbers you should get on our data
so there is something to check against.

All paths are relative to the project root, Jupyter is started from there, and
the kernel is `Python 3.11 (Computer Vision venv)`.

---

## 0. What the network actually returns

BODY_25 is a fully convolutional Caffe graph. It takes an image and returns a
`(1, 78, H/8, W/8)` tensor. The stride of 8 comes from the four poolings in the
VGG-like backbone: every output cell corresponds to an 8×8 square of the input.

Those 78 channels are three different things packed into one tensor:

| channels | what it is | how to read it |
|---|---|---|
| 0–24 | heatmaps of 25 joints | the value is the probability that the joint is here |
| 25 | background | not needed, but it takes up a slot |
| 26–77 | PAFs (Part Affinity Fields) | 26 limbs × 2 channels: the x and y components of a unit vector along the limb |

Heatmaps answer "where are the joints", PAFs answer "which of those joints belong
to the same person". The second question is exactly why PAFs exist: with three
people in frame, the left-wrist heatmap has three peaks, and without PAFs there is
no way to tell whose wrist is whose.

Two approaches to keep apart:

- **bottom-up** — run the whole frame once, find every joint, group them with
  PAFs. The time does not depend on the number of people.
- **top-down** — run a person detector first, then pose on each crop separately.
  The crop is stretched to the network input, so small people "grow"; you pay with
  time for every person.

---

## Block 1. Imports

```python
import os
import cv2
import numpy as np
import urllib.request          # ← not just urllib
import matplotlib.pyplot as plt
import time
%matplotlib inline
```

`import urllib` does not give you `urllib.request` — it is a separate submodule,
and without the explicit import you get an `AttributeError` exactly at download
time. `%matplotlib inline` makes `plt.show()` draw under the cell.

---

## Block 2. The model: a resumable download with a size check

**What it does:** fetches the architecture (`pose_deploy.prototxt`, ~42 KB) and
the weights (`pose_iter_584000.caffemodel`, exactly 104,715,850 bytes) into
`models/pose_body25/`.

**Why not a one-line `urlretrieve`:**

1. The official CMU host (`posefs1.perception.cs.cmu.edu`) has not resolved for a
   long time — every tutorial that uses it is dead. The architecture comes from
   CMU's GitHub, the weights from a Hugging Face mirror.
2. 100 MB over an unstable connection breaks halfway through. On a break,
   `urlretrieve` leaves a **truncated file that looks perfectly normal**:
   `os.path.isfile()` says True, the next run skips the download, and
   `readNetFromCaffe` fails with an unreadable protobuf parsing error. That is
   exactly what bit us: the folder held a 2.1 MB file instead of 104.7 MB.

**What your function has to be able to do:**

- send a `Range: bytes=<how much is already there>-` header and append to the file
  in `"ab"` mode;
- check that the server answered **206**, not 200 — a 200 means it ignored the
  Range and is sending the file from the start, so appending would corrupt it:
  reset the counter and open `"wb"`;
- catch the exception on every attempt and try again (a broken connection is
  normal here, not an emergency);
- verify the final size against a known number, not against the fact that the
  file exists.

```python
MODEL_DIR = "models/pose_body25"
protoFile   = os.path.join(MODEL_DIR, "pose_deploy.prototxt")
weightsFile = os.path.join(MODEL_DIR, "pose_iter_584000.caffemodel")

PROTO_URL = ("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
             "openpose/master/models/pose/body_25/pose_deploy.prototxt")
WEIGHTS_URL  = "https://huggingface.co/dylanholmes/openpose-caffemodels/resolve/main/body25.caffemodel"
WEIGHTS_SIZE = 104_715_850     # anything smaller is a broken download


def download(url, dst, expected_size=None, attempts=10, timeout=60):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    for attempt in range(1, attempts + 1):
        have = os.path.getsize(dst) if os.path.isfile(dst) else 0
        if expected_size and have >= expected_size:
            return True
        if expected_size is None and have:
            return True

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                total = have + int(resp.headers.get("Content-Length", 0))
                reported = have
                with open(dst, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if have - reported >= 10 << 20:        # report every 10 MB, because
                            reported = have                    # \r does not erase a line in Jupyter
                            print(f"  {os.path.basename(dst)}: {have/1e6:6.1f} / {total/1e6:.1f} MB")
        except Exception as exc:
            print(f"\n  attempt {attempt} broke off: {exc}")
            time.sleep(2)

    return os.path.isfile(dst) and (not expected_size or os.path.getsize(dst) >= expected_size)
```

**Trap:** `print(..., end="")` with `\r` does not overwrite the line in Jupyter the
way it does in a terminal — every call becomes its own output line. Hence a report
every 10 MB instead of every megabyte.

---

## Block 3. Loading the network

```python
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

The argument order in `readNetFromCaffe` is **prototxt first, then the weights**
(`readNetFromTensorflow` is the other way round: the `.pb` first). Swap them and
you get an error about an unparsed protobuf, similar to the one a truncated file
produces.

CPU here is not a compromise: there is no CUDA on a Mac, `cv2.dnn` does not use
Metal, and the OpenCL target is usually slower than the CPU for this network.

**Check:** `len(net.getLayerNames())` → **261**.

---

## Block 4. The BODY_25 constants

Three lists, and all three have to agree with each other.

**Joints** (the index is the heatmap channel):

```
0 Nose      5 LShoulder  10 RKnee    15 REye   20 LSmallToe
1 Neck      6 LElbow     11 RAnkle   16 LEye   21 LHeel
2 RShoulder 7 LWrist     12 LHip     17 REar   22 RBigToe
3 RElbow    8 MidHip     13 LKnee    18 LEar   23 RSmallToe
4 RWrist    9 RHip       14 LAnkle   19 LBigToe 24 RHeel
```

R/L are from the point of view of **the person in the frame**, not the viewer.

**`POSE_PAIRS`** — the 26 limbs the network computes PAFs for. **`MAP_IDX`** — for
each limb, the two PAF channels relative to `PAF_OFFSET = 26`. Both lists come from
`poseParameters.cpp` in OpenPose and their order is rigidly linked: `MAP_IDX[k]`
are the channels for exactly `POSE_PAIRS[k]`. Reorder one of them and the skeleton
will connect a knee to an ear — a silent failure, with no exception.

```python
POSE_PAIRS = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9],
              [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15],
              [15, 17], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19], [19, 20],
              [14, 21], [11, 22], [22, 23], [11, 24]]

MAP_IDX = [[0, 1], [14, 15], [22, 23], [16, 17], [18, 19], [24, 25], [26, 27],
           [6, 7], [2, 3], [4, 5], [8, 9], [10, 11], [12, 13], [30, 31],
           [32, 33], [36, 37], [34, 35], [38, 39], [20, 21], [28, 29], [40, 41],
           [42, 43], [44, 45], [46, 47], [48, 49], [50, 51]]

SKIP_RENDER = {18, 19}   # pairs [2,17] and [5,18] (shoulder-ear): useful for
                         # grouping, but they draw right across the face
```

**How to verify you did not mix them up:** draw the skeleton on
`data/images/messi5.jpg` and look at the anatomy — forearms should come out of
elbows, shins out of knees. That is more reliable than comparing numbers by eye.

You will also need `COLORS` — 26 BGR colors, one per limb — and a helper:

```python
def show(im, title="", figsize=(11, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(im[:, :, ::-1])      # BGR → RGB for matplotlib
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
```

---

## Block 5. A single forward pass

```python
def run_pose(net, im, in_height=368):
    h, w = im.shape[:2]
    in_width = max(8, int(round(in_height * w / h / 8)) * 8)

    blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (in_width, in_height),
                                 (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()                       # (1, 78, in_height/8, in_width/8)

    return np.stack([cv2.resize(out[0, i], (w, h)) for i in range(out.shape[1])])
```

Going through the arguments — every one of them has a reason:

- `1/255.0` — the model was trained on inputs in the [0, 1] range;
- `(in_width, in_height)` — the width is computed from the frame aspect ratio and
  rounded to a multiple of 8. Do not preserve the ratio and the person gets
  squeezed horizontally, which costs accuracy; do not round and OpenCV pads the
  size itself, shifting the maps by half a cell;
- `(0, 0, 0)` — no mean subtraction, the model does not expect it;
- **`swapRB=False`** — the Caffe model was trained on BGR, and `cv2.imread`
  returns exactly BGR. Set it to `True` (as in the TensorFlow models from the
  detection notebook) and the network still finds something, but worse and
  unstably. A classic silent bug;
- `crop=False` — otherwise OpenCV center-crops the frame instead of resizing it.

The last line stretches all 78 maps back to the frame size. That way peak
coordinates are already in original-image pixels and there is no need to multiply
by `w/out_w` in every function below. You pay with memory: 78 float maps at frame
resolution.

**Check on `messi5.jpg` (548×342):** `~0.9–1.2 s`, `maps.shape == (78, 342, 548)`.

---

## Block 6. Looking at what is inside

Not needed for the work itself, but useful for seeing what you are dealing with:
overlay `maps[i]` on the frame with `plt.imshow(..., alpha=0.6, cmap="jet")`. Take
`Nose`, `RWrist`, `LAnkle` — you will see compact hot spots.

For a PAF, draw the **magnitude** of the vector, since there are two channels:

```python
k = POSE_PAIRS.index([3, 4])                    # RElbow → RWrist
paf = np.hypot(maps[PAF_OFFSET + MAP_IDX[k][0]],
               maps[PAF_OFFSET + MAP_IDX[k][1]])
```

You will see a "stripe" along the forearm — that is the field we integrate along
later.

---

## Block 7. A single person

If there is exactly one person in the frame, PAFs are not needed: take the global
maximum of each heatmap.

```python
def keypoints_single(maps, threshold=0.1):
    points = []
    for i in range(N_POINTS):
        _, conf, _, loc = cv2.minMaxLoc(maps[i])
        points.append((loc[0], loc[1], conf) if conf > threshold else None)
    return points
```

`cv2.minMaxLoc` returns `(minVal, maxVal, minLoc, maxLoc)` — you need the second
and the fourth. The 0.1 threshold is deliberately low: a few false points are
better than losing a wrist in motion.

`draw_pose(im, points)` walks `POSE_PAIRS`, skips `SKIP_RENDER` and any pair where
one end is `None`, draws lines in `COLORS[k]` and circles at the points. Draw on a
copy (`im.copy()`), otherwise you corrupt the input frame — which hurts especially
in Jupyter, where cells are re-run selectively.

**Check:** on `messi5.jpg` it finds **24 of 25** joints (`RHeel` is missing — the
right heel is occluded).

**The limit of this approach:** two people in frame → one maximum per heatmap →
the joints of two bodies get merged into one mutant skeleton. Which is what the
next block is for.

---

## Block 8. All joint candidates

```python
def get_keypoints(prob_map, threshold=0.1):
    smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)
    mask = np.uint8(smooth > threshold)

    keypoints = []
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        blob_mask = cv2.fillConvexPoly(np.zeros(mask.shape), cnt, 1)
        _, _, _, loc = cv2.minMaxLoc(smooth * blob_mask)
        keypoints.append(loc + (prob_map[loc[1], loc[0]],))
    return keypoints
```

This is poor man's NMS: instead of searching for local maxima, threshold the map,
cut it into connected regions with `findContours`, and take one peak inside each
region. The 3×3 blur removes double peaks caused by noise.

Next, every point needs a global **id**, because grouping works on ids, not on
coordinates:

```python
def detect_keypoints(maps, threshold=0.1):
    detected, keypoints_list, next_id = [], [], 0
    for i in range(N_POINTS):
        with_id = []
        for p in get_keypoints(maps[i], threshold):
            keypoints_list.append(p[:3])       # (x, y, conf)
            with_id.append(p + (next_id,))     # (x, y, conf, id)
            next_id += 1
        detected.append(with_id)
    return detected, np.array(keypoints_list).reshape(-1, 3)
```

`detected[i]` holds the candidates for joint i; `keypoints_list` holds the same
points in one array, where the id is simply the row index. `.reshape(-1, 3)` saves
you on an empty frame: without it `np.array([])` has shape `(0,)` and indexing
blows up.

Watch the order: `loc` is `(x, y)`, while array indexing is `[y, x]`. Mix them up
and the coordinates get transposed — on square images you will not even notice.

---

## Block 9. PAFs: which joints are actually connected

The core math of the whole method.

For limb `k` between candidates `pa` and `pb`:

1. the direction `d = (pb - pa) / |pb - pa|` — a unit vector;
2. take `n_interp = 10` points evenly spaced along the segment `pa → pb`;
3. read the PAF vector `(paf_x, paf_y)` at each point and take the dot product
   with `d` — the projection of the field onto the limb direction;
4. the limb counts as real if **most** of the points (more than
   `conf_threshold = 0.6`) have a projection above `paf_threshold = 0.2`;
5. among all `pb` for a given `pa`, keep the one with the highest mean projection.

```python
d = np.subtract(pb[:2], pa[:2])
norm = np.linalg.norm(d)
if norm < 1e-6:
    continue                                   # the two points coincide — division by zero
d = d / norm

xs = np.linspace(pa[0], pb[0], n_interp)
ys = np.linspace(pa[1], pb[1], n_interp)
vectors = [[paf_x[int(round(y)), int(round(x))],
            paf_y[int(round(y)), int(round(x))]] for x, y in zip(xs, ys)]
scores = np.dot(vectors, d)

if (np.sum(scores > paf_threshold) / n_interp) > conf_threshold and scores.mean() > best_score:
    best_score, best_j = scores.mean(), j
```

Why two thresholds instead of one on the mean: a single very hot patch easily
drags the mean up while the rest of the segment runs across the background. The
"most points above the threshold" condition rejects such accidental connections
between different people.

The function returns `valid_pairs` (for each limb, a list of `(id_a, id_b, score)`)
and `invalid` — the indices of limbs where one side had no candidates at all; skip
those in the next block.

---

## Block 10. Stitching skeletons together

```python
def group_people(valid_pairs, invalid, keypoints_list):
    people = np.empty((0, N_POINTS + 1))

    for k, (a, b) in enumerate(POSE_PAIRS):
        if k in invalid:
            continue
        for id_a, id_b, score in valid_pairs[k]:
            found = next((i for i, person in enumerate(people) if person[a] == id_a), -1)

            if found != -1:                       # joint a already belongs to someone — append b
                people[found][b] = id_b
                people[found][-1] += keypoints_list[int(id_b), 2] + score
            elif k < 17:                          # start a new person only from
                row = np.full(N_POINTS + 1, -1.0) # the "torso" pairs
                row[a], row[b] = id_a, id_b
                row[-1] = keypoints_list[[int(id_a), int(id_b)], 2].sum() + score
                people = np.vstack([people, row])

    return people
```

Row format: 25 joint ids (`-1` means missing) plus the total score at the end.

The algorithm is greedy: walk the limbs in `POSE_PAIRS` order and, if joint `a`
already belongs to someone, append `b` to them. The order of the list is not
accidental — it starts with `[1,8]` (neck→hip) and continues along the torso, so
the skeleton grows out from the center.

`k < 17` is the cutoff: pairs from 17 on are ears, toes and heels. Allow a person
to start from those and you get "people" made of two toes.

`draw_people` then draws only skeletons with at least `min_parts=4` joints — that
filter removes fragments built from two random points.

**Check on `data/images/solders.jpg`:** ~1.1 s, 41 joint candidates, 5 skeletons.

---

## Block 11. `in_height` — the main trade-off knob

The network stride of 8 pixels is fixed. A person 60 pixels tall in the frame,
after resizing to `in_height=256`, takes up ~30 pixels of input, i.e. ~4 output
cells — there is simply nowhere for them to leave a distinct peak on a heatmap.

Measurements on `data/images/basketball1.png` (two players, one large, one partly
out of frame):

| `in_height` | time | skeletons found |
|---|---|---|
| 256 | 0.23 s | 1 |
| 368 | 0.41 s | 1 |
| 512 | 0.89 s | 2 |

The time grows roughly quadratically, so 368 is the usual default and 512+ is a
deliberate choice for when people are small.

---

## Block 12. Top-down: detector plus pose on the crop

Instead of inflating the input for the whole frame — find people with a detector
and run pose on each crop separately. The detector is the same SSD MobileNet v2 as
in `deep_learning_based_object_detection.ipynb` (class 1 in COCO = person).

```python
def pose_top_down(im, in_height=368, det_threshold=0.4, pad=0.15, draw_boxes=True):
    out = im.copy()
    for x1, y1, x2, y2, score in detect_persons(im, det_threshold):
        m = int(pad * max(x2 - x1, y2 - y1))       # the detector box clips hands and feet
        cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
        cx2, cy2 = min(im.shape[1], x2 + m), min(im.shape[0], y2 + m)
        crop = im[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        points = keypoints_single(run_pose(net, crop, in_height))
        points = [None if p is None else (p[0] + cx1, p[1] + cy1, p[2]) for p in points]

        scale = max(1, (y2 - y1) // 60)            # thinner lines for small people
        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
        out = draw_pose(out, points, radius=scale + 1, thickness=scale)
    return out
```

Three things that are easy to forget:

1. **padding the box** — the detector gives a tight box, so hands and feet end up
   on the edge or outside it;
2. **shifting the coordinates back** — points from the crop have to be returned to
   the frame coordinate system (`+cx1`, `+cy1`), otherwise the skeleton is drawn in
   the top-left corner;
3. **line width derived from the box size** — on a 30×60 pixel pedestrian, circles
   of radius 4 merge into one blob.

Inside a crop there is only one person, so `keypoints_single` is enough — no PAFs
needed.

**Measurements on frame 300 of `videos/vtest.mp4`:** bottom-up on the whole frame —
0.52 s, and the small pedestrians fall apart into loose points; top-down — 0.92 s
and 7 clean skeletons. A clear illustration of when to use which approach.

---

## Block 13. Video

No magic — a loop over frames, but:

- `cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))` — if
  you take every `step`-th frame, the output fps has to be divided by `step`, or
  the video plays fast-forward;
- the size in `VideoWriter` is `(width, height)`, while `frame.shape` is
  `(height, width)`. Mix them up and the file is created but stays empty (0 bytes
  or a few hundred);
- `cap.release()` and `writer.release()` in `finally`, otherwise after an exception
  the file stays unfinished and locked;
- do not print progress on every frame (see the `\r` trap in block 2).

**Measurements:** 60 frames of `vtest.mp4` through top-down — 38.6 s (~0.64 s per
frame), the result lands in `outputs/pose-vtest.mp4`.

---

## Block 14. Webcam

An ordinary `cap.read()` → pose → `cv2.imshow` → `cv2.waitKey(1)` loop, exiting on
ESC (27) or `q`. The specifics:

- use `in_height=192` — otherwise it is 1–2 FPS and feels frozen;
- `cv2.flip(frame, 1)` — mirroring, which feels natural when moving in front of a
  camera;
- `cap.release()` and `cv2.destroyAllWindows()` in `finally` — otherwise, after
  interrupting the cell, the window keeps hanging around and the camera is never
  released;
- keep the call commented out in the notebook so that "Run All" does not open a
  window.

---

## Common traps, collected

| Symptom | Cause |
|---|---|
| `readNetFromCaffe` fails while parsing | a truncated `.caffemodel`, or the argument order swapped |
| The skeleton connects a knee to an ear | `MAP_IDX` does not match `POSE_PAIRS` |
| Points are transposed | confusion between `loc = (x, y)` and `arr[y, x]` |
| The pose "drifts" with no visible error | `swapRB=True` instead of `False` |
| Two people become one mutant skeleton | `keypoints_single` where PAFs are required |
| Small people are not found | `in_height` too small; or switch to top-down |
| The skeleton is drawn in the corner of the frame | the coordinate shift from the crop is missing |
| An empty mp4 | `(w, h)` swapped in `VideoWriter` |
| Hundreds of progress lines | `print(..., end="")` with `\r` in Jupyter |
| The output frame gets corrupted between cells | drawing on the original instead of a `.copy()` |

---

## Where to go next

- **Temporal smoothing:** the pose is computed per frame, so the skeleton
  "breathes" on video. The simplest fix is exponential smoothing of the
  coordinates between frames.
- **Tracking:** link skeletons across frames (by box IoU at the very least) — then
  you can measure the movements of a specific person instead of "someone in frame".
- **Faster models:** `models/` already holds `pose_estimation_mediapipe_2023mar.onnx`
  and `person_detection_mediapipe_2023mar.onnx` — those run in real time on the
  CPU, unlike BODY_25. Or YOLO-pose.
- **Applications:** joint angles via `arctan2` between limb vectors — the basis for
  counting exercise repetitions or checking posture.
