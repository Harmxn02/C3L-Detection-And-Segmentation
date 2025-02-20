"""
Helper functions (color detection, preprocessing)
"""

import cv2
import numpy as np
import webcolors
from transformers import pipeline
from PIL import Image


def closest_color(requested_color):
    """
    Finds the closest named color using webcolors.

    Args:
        requested_color (tuple): The RGB values of the color to match.

    Returns:
        str: The name of the closest color from the webcolors CSS3 color list.
    """
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def detect_color(frame, mask):
    """
    Detects the median color of the segmented region.

    Args:
        frame (np.ndarray): The original frame from the webcam.
        mask (np.ndarray): The mask for the segmented object.

    Returns:
        str: The closest color name from the mask region.
    """
    masked_pixels = frame[mask > 0]
    if masked_pixels.size == 0:
        return "Unknown"
    median_color = np.median(masked_pixels, axis=0)
    return closest_color(tuple(map(int, median_color)))


def load_segmentation_pipeline():
    """
    Loads and returns the segmentation pipeline for detecting clothes and hair.

    Returns:
        pipeline: The Hugging Face segmentation pipeline.
    """
    return pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes",
        device=0  # Use GPU 0
    )


def load_hair_color_pipeline():
    """
    Loads and returns the hair color classification pipeline.

    Returns:
        pipeline: The Hugging Face hair color classification pipeline.
    """
    return pipeline(
        "image-classification",
        model="enzostvs/hair-color",
        device=0  # Use GPU 0
    )


def preprocess_frame(frame):
    """
    Converts an OpenCV frame (BGR format) to a PIL image (RGB format) for model inference.

    Args:
        frame (np.ndarray): The captured frame in BGR format.

    Returns:
        pil_image (PIL.Image.Image): The frame converted to RGB format.
    """
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def get_detected_items(results, frame, segmentation_pipe, hair_color_pipe):
    """
    Processes the results from the segmentation model and detects the color of clothing and hair.

    Args:
        results (list): The segmentation results.
        frame (np.ndarray): The current frame captured from the webcam.
        segmentation_pipe (pipeline): The segmentation pipeline.
        hair_color_pipe (pipeline): The hair color classification pipeline.

    Returns:
        detected_items (list): A list of detected items and their corresponding colors.
        mask_overlay (np.ndarray): The frame with segmentation masks applied.
    """
    detected_items = []
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    # Convert frame to PIL image for model inference
    pil_image = preprocess_frame(frame)

    for result in results:
        label = result['label']
        mask = np.array(result['mask'])

        if label == "Hair":
            # Use hair classification model instead of median color detection
            hair_result = hair_color_pipe(pil_image)  # Pass PIL image instead of OpenCV frame
            color_name = hair_result[0]['label']
        else:
            # Detect color normally for other segmented items
            color_name = detect_color(frame, mask)

        # Assign a random color for the mask overlay
        overlay_color = [np.random.randint(100, 255) for _ in range(3)]
        mask_overlay[mask > 0] = overlay_color

        detected_items.append(f"{label} ({color_name})")

    return detected_items, mask_overlay
