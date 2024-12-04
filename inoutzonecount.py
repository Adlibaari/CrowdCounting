import cv2
import argparse
from ultralytics import YOLO
from sort import *
import supervision as sv
import numpy as np
from supervision import ColorPalette

# Write video with OpenCV - ensure file extension and codec match
output_filename = 'inoutzone.mp4'
frame_rate = 20
frame_size = (1280, 720)  # Default frame size, will adjust to actual frame size

# Define multiple polygons
polygons = [
    np.array([[1605, 248], [1621, 496], [1289, 504], [1281, 256]], np.int32),
    np.array([[658, 248], [649, 527], [1013, 533]], np.int32),
]

colors = ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(r"C:\Users\Barry\Documents\Uni\Projects\Object Tracking\People Counting\Zone Counting\vidp.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")
    tracker = sv.ByteTrack()
    
    # Set up polygon zones and annotators
    zones = [sv.PolygonZone(polygon=polygon, frame_resolution_wh=(1920, 1080)) for polygon in polygons]
    zone_annotators = [
        sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=4, text_thickness=8, text_scale=4)
        for index, zone in enumerate(zones)
    ]
    box_annotators = [sv.BoxAnnotator(color=colors.by_idx(index), thickness=4) for index in range(len(polygons))]
    
    enter_list = [[] for _ in range(len(zones))]  # One list for each zone
    leave_list = [[] for _ in range(len(zones))]
    current_in_zones = [set() for _ in range(len(zones))]

    # Ensure dimensions match the actual video frames
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    height, width = frame.shape[:2]
    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    while ret:
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [f"#{tracker_id} {result.names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]

        # Process each zone
        for i, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, box_annotators)):
            mask = zone.trigger(detections=detections)
            in_zone_detections = detections[mask]

            current_frame_in_zone = set()

            for detection in in_zone_detections:
                tracker_id = detection[4]
                current_frame_in_zone.add(tracker_id)

                if tracker_id not in enter_list[i]:
                    enter_list[i].append(tracker_id)
                    print(f"Object Entered Zone {i}: Tracker ID: {tracker_id}")

            objects_left_zone = current_in_zones[i] - current_frame_in_zone
            for tracker_id in objects_left_zone:
                if tracker_id not in leave_list[i]:
                    leave_list[i].append(tracker_id)
                    print(f"Object Left Zone {i}: Tracker ID: {tracker_id}")

            current_in_zones[i] = current_frame_in_zone

            # Annotate the frame
            frame = box_annotator.annotate(scene=frame, detections=in_zone_detections)
            frame = zone_annotator.annotate(scene=frame)

            # Set the vertical offset dynamically based on the zone index
            vertical_offset = 50 + i * 150  
            
            # Display enter/leave counts per zone with dynamic positioning
            cv2.putText(frame, f"Zone {i} Enter: {len(enter_list[i])}", (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Zone {i} Leave: {len(leave_list[i])}", (10, vertical_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video.write(frame)
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()  # Read the next frame

    # Release resources
    video.release()
    cap.release()
    cv2.destroyAllWindows()

    # Final output of enter/leave lists
    for i in range(len(zones)):
        print(f"Zone {i} - IDs entered: {enter_list[i]}")
        print(f"Zone {i} - IDs left: {leave_list[i]}")

if __name__ == "__main__":
    main()
