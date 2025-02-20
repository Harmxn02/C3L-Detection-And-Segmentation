"""
This Python script is used to detect the hair colour of a person using the device's webcam.
The script uses the Hugging Face Transformers library to load the pre-trained model for clothing detection.
The script uses the OpenCV library to capture the video feed from the device's webcam.
The script uses the webcolors library to find the closest named color to the detected color.
The script uses the PIL library to convert the OpenCV frame to a PIL image for the model.
"""

import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import webcolors

def closest_color(requested_color):
    """
    Finds the closest named color using webcolors.
    """
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def detect_color(frame, x1, y1, x2, y2):
    """
    Detects the median color of the region inside the bounding box.
    """
    height, width, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    median_color = np.median(roi_rgb.reshape(-1, 3), axis=0)
    return closest_color(tuple(map(int, median_color)))

pipe = pipeline(
    "object-detection",
    model="valentinafeve/yolos-fashionpedia",
    device=0  # Use GPU 0
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = pipe(pil_image)
    detections = [d for d in detections if d["score"] > 0.85]

    if detections:
        top_detection = max(detections, key=lambda d: d["score"])
        x1, y1, x2, y2 = int(top_detection["box"]["xmin"]), int(top_detection["box"]["ymin"]), int(top_detection["box"]["xmax"]), int(top_detection["box"]["ymax"])
        item = top_detection["label"]
        color_name = detect_color(frame, x1, y1, x2, y2)
        cv2.putText(frame, f"{item} ({color_name})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for detection in detections:
            print(f"Detected: {detection['label']} ({color_name})\n")

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
