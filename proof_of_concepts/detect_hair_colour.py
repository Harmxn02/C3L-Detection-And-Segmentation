"""
This Python script is used to detect the hair colour of a person using the device's webcam.
The script uses the Hugging Face Transformers library to load the pre-trained model for hair colour detection.
The script uses the OpenCV library to capture the video feed from the device's webcam.
The script uses the PIL library to convert the OpenCV frame to a PIL image for the model.
"""


import cv2
import numpy as np
import time
from datetime import datetime
from PIL import Image
from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="enzostvs/hair-color",
    device=0 # GPU
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame.")
        break
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = pipe(pil_image)

    cv2.putText(frame, result[0]['label'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(f"{result}\n")

    cv2.imshow('Hair Colour Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
