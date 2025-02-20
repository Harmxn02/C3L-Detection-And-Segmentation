"""
Handles webcam capture & processing
"""

import cv2
import numpy as np
from PIL import Image

def get_webcam_feed():
    """
    Opens the device's webcam and returns a video capture object.

    Returns:
        cap (cv2.VideoCapture): The webcam capture object.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open webcam.")
    
    return cap

def capture_frame(cap):
    """
    Captures a single frame from the webcam.

    Args:
        cap (cv2.VideoCapture): The webcam capture object.

    Returns:
        frame (np.ndarray): The captured frame from the webcam.
    """
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read frame.")
    
    return frame

def process_frame(frame):
    """
    Converts a captured frame (BGR format) to a PIL image (RGB format) for model processing.

    Args:
        frame (np.ndarray): The frame captured from the webcam in BGR format.

    Returns:
        pil_image (PIL.Image.Image): The converted PIL image in RGB format.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return pil_image
