from ultralytics import YOLO
import cv2
import time
import numpy as np
import yaml

data = [
    "waterbottle",
    "coke",
    "milk",
    "vitamin",
    "box",
    "table",
    "chair",
    "monitor",
    "laptop",
    "phone",
    "bed",
    "pillow",
    "keyboard",
    "mouse",
    "cabinet",
    "whiteboard",
    "trashcan",
    "cup",
    "backpack",
    "bag",
    "shoe",
    "headphone",
    "treadmill",
    "apple",
    "banana",
    "orange",
    "book",
    "vase",
    "scissors",
    "person",
    "remote"
]
model_coco = YOLO("yolov8s-seg.pt")
model_coco.predict(source=np.zeros((1, 1, 3)), conf=1, show=False)

model_10cls = YOLO("all2ndvid-10cls-best-seg.pt")
model_10cls.predict(source=np.zeros((1, 1, 3)), conf=1, show=False)

rand_color = np.random.rand(40, 3) * 255


def mod_predict(output, image, model, dataset="coco", allowed_class={"person": 0}, conf=0.5, show_result=False):
    result = model.predict(source=image, conf=conf, show=show_result)[0]
    if dataset == "coco":
        with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
            datasets_names = yaml.safe_load(stream)['names']
    else:
        datasets_names = dataset

    if result.masks:

        for (box, mask) in zip(result.boxes, result.masks.data):
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            name = datasets_names[int(box.cls.cpu().numpy())
                                  ] if datasets_names else 'unknown'
            # x, y = x-w/2, y-h/2

            if name in allowed_class:
                for alr in output:
                    if get_iou((x1, y1, x2, y2), alr[2]) > 0.7:
                        # print("Skipp", name)
                        break
                else:
                    output.append(
                        (name, allowed_class[name], (x1, y1, x2, y2), mask.cpu().numpy().astype(bool)))
            else:
                continue

    # return output


def draw_mask(frame, mask, id, opa=0.4):

    color = rand_color[id]
    frame[mask] = frame[mask] * (1-opa) + color * (opa)


def draw_box(frame, bbox, name, id):
    color = rand_color[id]
    x1, y1, x2, y2 = bbox

    # cv2.rectangle(frame, (int(x1), int(y1)),
    #               (int(x2), int(y2)), color, 2)
    cv2.putText(frame, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,
                1, color, 2)


def get_iou(bb1, bb2):
    b1x1, b1y1, b1x2, b1y2 = bb1
    b2x1, b2y1, b2x2, b2y2 = bb2
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert b1x1 < b1x2
    assert b1y1 < b1y2
    assert b2x1 < b2x2
    assert b2y1 < b2y2

    # determine the coordinates of the intersection rectangle
    x_left = max(b1x1, b2x1)
    y_top = max(b1y1, b2y1)
    x_right = min(b1x2, b2x2)
    y_bottom = min(b1y2, b2y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
    bb2_area = (b2x2 - b2x1) * (b2y2 - b2y1)
    iou = intersection_area / \
        float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


start = time.time()
cap = cv2.VideoCapture(int(input("cam: ")))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    drawn_frame = np.copy(frame)
    print(frame.shape)
    output = []

    mod_predict(output, frame, model_10cls, dataset={
                0: "apple", 1: "banana", 2: "coke", 3: "keyboard", 4: "laptop", 5: "milk", 6: "mouse", 7: "orange", 8: "vitamin", 9: "waterbottle"}, allowed_class={"apple": 23, "banana": 24, "orange": 25, "waterbottle": 0, "coke": 1, "milk": 2, "vitamin": 3})
    mod_predict(output, frame, model_coco, dataset="coco", allowed_class={
                "dining table": 5, "chair": 6, "tv": 7, "handbag": 19, "backpack": 18, "bottle": 0, "cup": 17, "bed": 10, "tv": 7, "laptop": 8, "mouse": 13, "keyboard": 12, "cell phone": 9, "refrigerator": 14, "book": 26, "vase": 27, "scissors": 28, "person": 29, "remote": 30})

    # # if results.boxes:
    # #     for obj in results.boxes:
    # #         # print(obj.xywhn)
    # #         # print(obj.cls)
    # #         # print(obj.conf)
    # #         print(obj.data)

    # if results.masks:
    #     # print(len(results.masks.segments))
    #     # for obj in results.masks.segments:
    #     #     print(obj)
    #     #     # print(obj.data)
    #     # print(results.masks.data.shape)

    #     for i, obj in enumerate(results.masks.data):
    #         frame[obj.cpu().numpy().astype(bool)] = frame[obj.cpu(
    #         ).numpy().astype(bool)] * 0.6 + np.array((0, 120, 0)) * 0.4

    # # if results.probs:
    # #     print(results.probs)

    for obj in output:
        name, id, bbox, mask = obj

        draw_mask(drawn_frame, mask, id)
        draw_box(drawn_frame, bbox, data[id], id)
        print(name, id, bbox)

    # cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    start = time.time()

    # cv2.imshow("frame", frame)
    cv2.imshow("drawn_frame", cv2.resize(drawn_frame, (960, 720)))

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
