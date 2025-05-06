from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt nếu máy đủ mạnh

def count_people(image):
    results = model(image)
    count = 0
    for r in results:
        for c in r.boxes.cls:
            if int(c) == 0:  # class 0 = 'person'
                count += 1
    return count
