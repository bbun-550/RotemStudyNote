# pip install ultralytics

from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"error : {e}")
    

print(model.names)   # COCO dataset(class 80개)