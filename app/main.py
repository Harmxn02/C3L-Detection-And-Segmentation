import streamlit as st
import cv2
from webcam import get_webcam_feed, capture_frame, process_frame
from detector import run_detection

def display_frame(frame, detected_items):
    """
    Displays the frame with the detected items overlayed on it in Streamlit.

    Args:
        frame (np.ndarray): The frame with detection overlays.
        detected_items (list): List of detected items with their associated colors.
    """
    # Convert the frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    st.image(frame_rgb, caption="Webcam Feed", use_container_width=True)

    # Display the detected items
    if detected_items:
        st.write("Detected Items:")
        for item in detected_items:
            st.write(item)

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

    # Streamlit loop to capture frames and run detection
    stframe = st.empty()
    
    while st.session_state.webcam_running:
        frame = capture_frame(cap)

        # Run detection on the captured frame
        detected_items, segmented_frame = run_detection(frame)

        # Display the frame and detected items in the Streamlit interface
        display_frame(segmented_frame, detected_items)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
