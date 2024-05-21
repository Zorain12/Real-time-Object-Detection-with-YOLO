# Install Libraries
# pip install opencv-python
# pip install ultralytics

# Import Libraries
import cv2
import numpy as np
import random
from ultralytics import YOLO

# Load YOLOv8 Model for object detection
model = YOLO("yolov8n.pt") 


# Generate unique colors
def unique_color(exist_colors):
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color not in exist_colors and not shade_of_red(color):
            return color


def shade_of_red(color):
    r, g, b = color
    return r > 100 and g < 100 and b < 100


# Bounding boxes with unique colors
capture = cv2.VideoCapture("rtsp://192.168.10.4:8080/h264_ulaw.sdp")  # RTSP video stream
colors = {}
timer_start = False
select_id = None
timer = 0


# User interaction
def click_event(event, x, y, flags, param):
    global select_id, timer, colors, timer_start
    if event == cv2.EVENT_LBUTTONDOWN:
        for person in people:
            bounding_box = person["bounding_box"]
            id = person["id"]
            if bounding_box[0] < x < bounding_box[2] and bounding_box[1] < y < bounding_box[3]:
                if select_id is not None:
                    colors[select_id] = unique_color(colors.values())
                select_id = id
                timer = 0
                timer_start = False
                break


cv2.namedWindow("Object_detection")
cv2.setMouseCallback("Object_detection", click_event)

while True:
    # Reads a frame from the video stream
    return_val, frame = capture.read()              
    if not return_val:
        break
    
    # Object Detection
    results = model(frame)

    # Processing Detections
    people = []
    for result in results:
        for detection in result.boxes:
            cls = detection.cls.cpu().numpy()
            if cls == 0:  # Class 0 is usually "person" in Common objects dataset
                bounding_box = detection.xyxy.cpu().numpy()[0]
                people.append({"bounding_box": bounding_box, "id": len(people)})

    # Drawing Bounding Boxes
    for person in people:
        bounding_box = person["bounding_box"]
        id = person["id"]
        if id not in colors:
            colors[id] = unique_color(colors.values())

        color = colors[id]
        if id == select_id:
            color = (0, 0, 255)  # Red color for the selected bounding box
            timer_start = True

        cv2.rectangle(
            frame, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), color, 2
        )
        if id == select_id and timer_start:
            cv2.putText(
                frame,
                f"Timer: {timer}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # Displaying Frame
    cv2.imshow("Object_detection", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):  #  press'q' key to exit the loop.
        break

    if timer_start:
        timer += 1

capture.release()
cv2.destroyAllWindows()