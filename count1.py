import cv2
from ultralytics import YOLO
from collections import defaultdict


model = YOLO("best.pt")

# Vehicle classes
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Open video
cap = cv2.VideoCapture("video4.mp4")

# Frame size
frame_width = 1020
frame_height = 600

# Line position for counting
line_y = 350

# Track vehicle IDs
vehicle_ids = set()


vehicle_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):

            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(class_id)]

            if label in vehicle_classes:

                # Bounding box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,255,255),2)

                
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

                
                if line_y-5 < cy < line_y+5:

                    if track_id not in vehicle_ids:

                        vehicle_ids.add(track_id)
                        vehicle_count += 1

    # Draw counting line
    cv2.line(frame,(0,line_y),(frame_width,line_y),(255,0,0),3)

    # Display vehicle count
    cv2.putText(frame,
                "Vehicle Count: "+str(vehicle_count),
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,255,255),
                3)

    cv2.imshow("Vehicle Detection and Counting",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()