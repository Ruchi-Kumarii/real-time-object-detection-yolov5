import cv2
import torch
import numpy as np
from torchvision.ops import nms
import time


# 1. DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. LOAD YOLOv5 MODEL

model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
     pretrained=True #Load weights already trained on a dataset
)
model.to(device)
model.eval()


# 3. CONFIG

CONF_THRESH = 0.6
IOU_THRESH = 0.5

ALLOWED_CLASSES = ["person", "bottle", "cup", "cell phone"]


# 4. WEBCAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Press 'q' to quit")


# 5. MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # ---------------- YOLO INFERENCE ----------------
    with torch.no_grad():
        results = model(frame)

    detections = results.xyxy[0]
    if detections is None or len(detections) == 0:
        cv2.imshow("Object Detection & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    detections = detections.cpu()

    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5]

    # ---------------- CONFIDENCE FILTER ----------------
    mask = scores > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    # ---------------- NMS ----------------
    keep = nms(boxes, scores, IOU_THRESH)
    boxes = boxes[keep].int().numpy() #boxex are pytorch tensor int converts coordinates to integer
    scores = scores[keep].numpy()
    classes = classes[keep].numpy()

    
    # COUNTING LOGIC
    
    person_count = 0
    presence = {
        "bottle": 0,
        "cup": 0,
        "cell phone": 0
    }

    # ---------------- DRAW BOXES & COUNT ----------------
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = model.names[int(classes[i])]
        conf = scores[i]

        if label not in ALLOWED_CLASSES:
            continue

        # Person → per-frame count
        if label == "person":
            person_count += 1
        else:
            presence[label] = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    
    # DISPLAY COUNTS (ONLY IF > 0)
    
    y = 60

    if person_count > 0:
        cv2.putText(
            frame,
            f"person: {person_count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 30

    for cls, val in presence.items():
        if val > 0:
            cv2.putText(
                frame,
                f"{cls}: {val}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            y += 30

    # ---------------- FPS ----------------
    fps = int(1 / max(time.time() - start_time, 1e-5))
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Object Detection & Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP

cap.release()
cv2.destroyAllWindows()
