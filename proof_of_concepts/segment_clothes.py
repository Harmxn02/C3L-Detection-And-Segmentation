import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import webcolors
import random

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

def detect_color(frame, mask):
    """
    Detects the median color of the segmented clothing region.
    """
    masked_pixels = frame[mask > 0]
    if masked_pixels.size == 0:
        return "Unknown"
    median_color = np.median(masked_pixels, axis=0)
    return closest_color(tuple(map(int, median_color)))

# Excluded classes from detection
EXCLUDED_CLASSES = {"Background"}

# Define segmentation pipeline
pipe = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b2_clothes",
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
    results = pipe(pil_image)
    
    detected_items = []
    
    # Overlay segmented objects
    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    for result in results:
        label = result['label']
        if label in EXCLUDED_CLASSES:
            continue
        
        mask = np.array(result['mask'])
        color_name = detect_color(frame, mask)

        # Assign a random color for the mask
        overlay_color = [random.randint(100, 255) for _ in range(3)]

        # Apply the mask overlay
        mask_overlay[mask > 0] = overlay_color

        detected_items.append(f"{label} ({color_name})")

    # Blend the original frame with the mask overlay
    alpha = 0.5  # Transparency factor for the overlay
    segmented_frame = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)

    # Display detected items
    if detected_items:
        text = " - ".join(detected_items)
        cv2.putText(segmented_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Detected: {text}\n")

    # Show segmented results
    cv2.imshow("Clothing Segmentation", segmented_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
