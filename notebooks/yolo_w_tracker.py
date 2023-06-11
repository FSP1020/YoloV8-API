from ultralytics import YOLO
import cv2 as cv
from ultralytics.tracker.trackers.byte_tracker import BYTETracker

coco_classes = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

class args:
    def __init__(self):
        self.tracker_type = 'bytetrack'
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8


model = YOLO("../models/weights/yolov8n.pt")

tracker_args = args()

tracker = BYTETracker(tracker_args, frame_rate=30)
tracking = True

camera = cv.VideoCapture(0)

while True:
    ret, frame = camera.read()

    results = model(frame, verbose=False)
    results = results[0].boxes.cpu().numpy()

    if len(results) != 0:
        results = tracker.update(results, frame)

        for detection in results:
            if tracking:
                bbox = detection[:4]
                id = int(detection[4])
                class_id = int(detection[6])
                class_name = coco_classes[class_id]
                
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Draw the bounding box
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                
                # Calculate the size of the label box based on the text
                text_width, text_height = cv.getTextSize("id:" + str(id) + " - " + class_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_box_width = text_width + 10
                label_box_height = text_height + 10
                
                # Draw the label box within the bounding box
                cv.rectangle(frame, (x1, y1), (x1 + label_box_width, y1 + label_box_height), (255, 0, 0), -1)
                
                # Write the class name inside the label box
                cv.putText(frame, "id:" + str(id) + " - " + class_name, (x1 + 5, y1 + text_height + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                bbox = detection.boxes.numpy()[0]
                class_id = int(bbox[5])
                class_name = coco_classes[class_id]
                
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Draw the bounding box
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                
                # Calculate the size of the label box based on the text
                text_width, text_height = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_box_width = text_width + 10
                label_box_height = text_height + 10
                
                # Draw the label box within the bounding box
                cv.rectangle(frame, (x1, y1), (x1 + label_box_width, y1 + label_box_height), (255, 0, 0), -1)
                
                # Write the class name inside the label box
                cv.putText(frame, class_name, (x1 + 5, y1 + text_height + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Display the frame in a window called "Webcam"
    cv.imshow('Webcam', frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the window
camera.release()
cv.destroyAllWindows()
