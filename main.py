from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils.torch_utils import select_device
import torch
import yaml


model = YOLO("yolov8s-seg.pt")

with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
    try:
        datasets = yaml.safe_load(stream)
        datasets_names = datasets['names']
    except:
        print("No file found")
        datasets_names = ""


start = time.time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()

    results = model.predict(source=frame, conf=0.5, show=True)[0]
    if results.boxes:
        # print(f"DETECT {len(results.boxes)}")
        output = dict()
        for i, obj in enumerate(results.boxes):
            x,y,w,h = obj.xywhn.cpu().numpy()[0]
            name = datasets_names[int(obj.cls.cpu().numpy())
                                  ] if datasets_names else 'unknown'
            output[i] = [name, x, y, w, h]

    print(output)

    # cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
