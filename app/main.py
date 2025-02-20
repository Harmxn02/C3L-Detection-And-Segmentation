import streamlit as st
import cv2
from webcam import get_webcam_feed, capture_frame, process_frame
from detector import run_detection

def display_frame(frame, detected_items, webcam_placeholder, log_placeholder):
    """
    Displays the frame with the detected items overlayed on it in Streamlit.

    Args:
        frame (np.ndarray): The frame with detection overlays.
        detected_items (list): List of detected items with their associated colors.
        webcam_placeholder (st.empty): Placeholder for webcam feed.
        log_placeholder (st.empty): Placeholder for logs.
    """
    # Update the webcam feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    webcam_placeholder.image(frame_rgb, caption="Webcam Feed", use_container_width=True)

    # Update the detected items log
    log_placeholder.text("Detected Items:")
    for item in detected_items:
        log_placeholder.text(item)

def main():
    st.title("Hair & Clothing Color Detection")

    # Initialize session state for webcam stop flag
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = True

    # Request permissions for webcam access
    st.sidebar.header("Webcam Access")
    st.sidebar.text("Allow access to your webcam to detect hair and clothing colors.")

    # Initialize webcam feed
    try:
        cap = get_webcam_feed()
    except ValueError as e:
        st.error(str(e))
        return

    # Create placeholders for webcam feed and logs
    webcam_placeholder = st.empty()  # Placeholder for webcam feed
    log_placeholder = st.empty()  # Placeholder for detected items

    # Streamlit loop to capture frames and run detection
    while st.session_state.webcam_running:
        frame = capture_frame(cap)

        # Run detection on the captured frame
        detected_items, segmented_frame = run_detection(frame)

        # Display the frame and detected items in the Streamlit interface
        display_frame(segmented_frame, detected_items, webcam_placeholder, log_placeholder)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
