import numpy as np
from ultralytics import YOLO
import cv2
from sort import *
import datetime

# Load the video file
cap = cv2.VideoCapture("path/to/video")

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get input video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the video writer
video = cv2.VideoWriter(
    'linecount.avi',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,  # Match the input video's frame rate
    (frame_width, frame_height)  # Match input video's resolution
)

# Load the YOLO model
model = YOLO("path/to/model")

# Class names for YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize the tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting limits
limitsDown = [103, 161, 550, 161]  # Line for counting people going down
limitsUp = [103, 200, 550, 200]  # Line for counting people going up

# Variables to track IDs and their dwell time
totalCountUp = []
totalCountDown = []
object_id_list = []
dtime = dict()
track_history = dict()

while True:
    success, img = cap.read()  # Read a frame
    if not success:
        break  # Break the loop if no frame is returned (end of video)

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    # Process YOLO detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with detections
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, objectId = map(int, result)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box around the detected object
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Calculate dwell time
        if objectId in object_id_list:
            curr_time = datetime.datetime.now()
            old_time = dtime[objectId]
            time_diff = curr_time - old_time
            dtime[objectId] = curr_time
            sec = time_diff.total_seconds()
            new_time = track_history[objectId][-1][4] + sec
            track_history[objectId].append([x1, y1, x2, y2, new_time])
            if len(track_history[objectId]) > 30:
                track_history[objectId].pop(0)
        else:
            object_id_list.append(objectId)
            dtime[objectId] = datetime.datetime.now()
            track_history[objectId] = [[x1, y1, x2, y2, 0]]

        # Display the object ID and dwell time
        text = f"{objectId}|{int(track_history[objectId][-1][4])}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw the object's centroid
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Draw the track history
        for i in range(1, len(track_history[objectId])):
            prev_x, prev_y = track_history[objectId][i - 1][0:2]
            curr_x, curr_y = track_history[objectId][i][0:2]
            cv2.line(img, (prev_x + w // 2, prev_y + h // 2),
                     (curr_x + w // 2, curr_y + h // 2), (255, 0, 255), 2)

        # Check if the object crosses the counting lines
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if objectId not in totalCountUp and objectId not in totalCountDown:
                totalCountUp.append(objectId)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        elif limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if objectId not in totalCountDown and objectId not in totalCountUp:
                totalCountDown.append(objectId)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 7)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 7)

    # Write the processed frame to the output video
    video.write(img)

    # Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video.release()
cv2.destroyAllWindows()

