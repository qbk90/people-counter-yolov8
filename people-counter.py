import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

# Predicto model instance creation
model = YOLO("yolov8s.pt")  # yolov8s (small) model

model.to("cuda")  # Run model on GPU

# Areas of interest coordinates
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]


# Function to get the coordinates of the mouse cursor when
# the cursor is inside the window
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cursor_coordinates = [x, y]
        print(cursor_coordinates)


cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_click)

cap = cv2.VideoCapture("peoplecount1.mp4")

# Classes of interest list creation
classes_file = open("classes.txt", "r")
classes_list = classes_file.read().split("\n")

# print(classes_list)

# Counter for frame skipping
counter = 0

# Create instance of Tracker class
tracker = Tracker()
while True:
    # reads a frame, ret is false when there's no more
    # frames left
    ret, frame = cap.read()
    if not ret:
        break
    counter += 1
    if counter % 2 != 0:
        # Skips every other frame
        continue
    frame = cv2.resize(frame, (1020, 500))
    # frame=cv2.flip(frame,1)

    # Store the predicted objects data into a
    # Pandas dataframe
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    peoples_list = []

    for index, row in px.iterrows():
        # If the detected object is a person,
        # draw a rectangle around them

        # print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        obj_class = int(row[5])
        obj_class_name = classes_list[obj_class]
        if obj_class_name == "person":
            peoples_list.append([x1, y1, x2, y2])
            tracked_people = tracker.update(peoples_list)
            for tracked_person in tracked_people:
                x3, y3, x4, y4, person_id = tracked_person
                # if the object detected is a person
                test_result = cv2.pointPolygonTest(
                    np.array(area2, np.int32), (x2, y2), False
                )
                if test_result >= 1:
                    # test if detected object is inside area 2
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{obj_class_name} {person_id}",
                        (x3, y3),
                        cv2.FONT_HERSHEY_DUPLEX,
                        (0.5),
                        (255, 255, 255),
                        1,
                    )
                    # draw a circle in the bottom right corner of the detected object (person)
                    cv2.circle(frame, (x4, y4), 3, (255, 0, 255), -1)

    # Draw the areas of interest
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(
        frame, str("1"), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1
    )

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(
        frame, str("2"), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1
    )

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
