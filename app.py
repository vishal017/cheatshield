import streamlit as st
import cv2
import numpy as np
from src.face_detection import detect_faces
from src.eye_blink_detection import load_eye_blink_model, detect_eye_blink
from src.object_detection import detect_objects
from src.audio_detection import detect_background_noise
from src.head_pose_detection import detect_head_pose
from src.system_monitoring import monitor_system

# Load models
eye_blink_model = load_eye_blink_model()

# Streamlit App
st.title("AI-Based Proctoring System")
st.write("Welcome to the AI-Based Proctoring System for Cheating Prevention in Online Exams.")

# Start Proctoring
if st.button("Start Proctoring"):
    st.write("Proctoring started...")
    video_capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Run face detection
        frame, faces_detected = detect_faces(frame)

        # Run eye blink detection
        frame, eye_blink_detected = detect_eye_blink(frame, eye_blink_model)

        # Run object detection
        frame, objects_detected = detect_objects(frame)

        # Run head pose detection
        frame = detect_head_pose(frame)

        # Display the frame with annotations
        frame_placeholder.image(frame, channels="BGR")

        # Display warnings
        if faces_detected > 1:
            st.warning("Multiple faces detected!")
        if eye_blink_detected:
            st.warning("Eye blink anomaly detected!")
        if objects_detected:
            st.warning("Prohibited objects detected!")

        # Stop proctoring
        if st.button("Stop Proctoring"):
            break

    video_capture.release()