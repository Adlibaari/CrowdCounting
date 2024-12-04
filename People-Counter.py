import mysql.connector
import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *
import datetime

# Function to get new session ID
def get_new_session_id(cursor):
    cursor.execute("SELECT MAX(session) FROM person_tracking")
    result = cursor.fetchone()[0]
    if result is None:
        return 1
    else:
        return result + 1

# MySQL connection setup
mydb = mysql.connector.connect(
    host="127.0.0.1",  # replace with your database host
    user="root",       # replace with your database username
    password="Solum23.",  # replace with your database password
    database="testdb"  # replace with your database name
)

mycursor = mydb.cursor()

# Get new session ID for the current run
session_id = get_new_session_id(mycursor)

cap = cv2.VideoCapture("vidp.mp4")  # For Video
model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", ...]  # Your class names
mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsDown = [103, 161, 500, 161]  # Your predefined limits
limitsUp = [103, 200, 500, 200]

object_id_list = []
dtime = dict()
track_history = dict()

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes 
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0]) 
            currentClass = classNames[cls]
        
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, objectId = map(int, result)

        if objectId in object_id_list:
            curr_time = datetime.datetime.now()
            old_time = dtime[objectId]
            time_diff = curr_time - old_time
            dtime[objectId] = datetime.datetime.now()
            sec = time_diff.total_seconds()
            new_time = track_history[objectId][-1][4] + sec
            track_history[objectId].append([x1, y1, x2, y2, new_time])

            # Insert the data into MySQL
            sql = "INSERT INTO person_tracking (session, personID, x1, y1, x2, y2, DwellTime) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (session_id, objectId, x1, y1, x2, y2, new_time)
            mycursor.execute(sql, val)
            mydb.commit()

        else:
            object_id_list.append(objectId)
            dtime[objectId] = datetime.datetime.now()
            track_history[objectId] = [[x1, y1, x2, y2, 0]]

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the database connection
mydb.close()
