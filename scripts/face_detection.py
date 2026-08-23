import cv2
import sys
from pathlib import Path

# Path relative to the script location, not to the current working directory:
# otherwise running from PyCharm, the terminal and Jupyter would each resolve it differently.
DATA_DIR = Path(__file__).parent.parent / "data"

video_source = 0
if len(sys.argv) > 1:
    video_source = sys.argv[1]

source = cv2.VideoCapture(video_source)

window_name = 'Camera Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

face_detector = cv2.dnn.readNetFromCaffe(
    str(DATA_DIR / "deploy.prototxt"),
    str(DATA_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"),
)

input_width = 300
input_height = 300
mean_subtraction = [104,117,123]
confidence_threshold = 0.1

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    input_blob = cv2.dnn.blobFromImage(frame, 1.0, (input_width, input_height), mean_subtraction, swapRB = False, crop = False)

    face_detector.setInput(input_blob)
    detections = face_detector.forward()

    for detection_index in range(detections.shape[2]):
        confidence = detections[0, 0, detection_index, 2]
        if confidence > confidence_threshold:
            x_left_bottom = int(detections[0,0,detection_index,3] * frame_width)
            y_left_bottom = int(detections[0,0,detection_index,4] * frame_height)
            x_right_top = int(detections[0,0,detection_index,5] * frame_width)
            y_right_top = int(detections[0,0,detection_index,6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label_text = "Confidence: %.4F" % confidence
            label_text_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_text_size[1]),
            (x_left_bottom + label_text_size[0], y_left_bottom + base_line),
            (255,255,255), cv2.FILLED)

            cv2.putText(frame, label_text, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    inference_ticks, _ = face_detector.getPerfProfile()
    label_text = 'Inference time: %.2f ms' % ( inference_ticks * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label_text, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(window_name, frame)