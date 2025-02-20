"""
Handles model inference (imports models & runs them)
"""

from transformers import pipeline
from utils import load_segmentation_pipeline, load_hair_color_pipeline, get_detected_items, preprocess_frame
import cv2

def run_detection(frame):
    """
    Runs segmentation and color detection on the provided frame.

    Args:
        frame (np.ndarray): The captured frame from the webcam.

    Returns:
        detected_items (list): List of detected items and their associated colors.
        segmented_frame (np.ndarray): The frame with segmentation overlays applied.
    """
    # Load the segmentation and hair color pipelines
    segmentation_pipe = load_segmentation_pipeline()
    hair_color_pipe = load_hair_color_pipeline()

    # Preprocess the frame to convert it into a format that the model can use
    pil_image = preprocess_frame(frame)

    # Run segmentation on the image
    results = segmentation_pipe(pil_image)

    # Get the detected items and overlay mask
    detected_items, mask_overlay = get_detected_items(results, frame, segmentation_pipe, hair_color_pipe)

    # Blend the original frame with the mask overlay
    alpha = 0.5  # Transparency factor for the overlay
    segmented_frame = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)

    return detected_items, segmented_frame
