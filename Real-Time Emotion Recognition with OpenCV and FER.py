import cv2
import streamlit as st
from fer import FER
from PIL import Image
import numpy as np

emotion_detector = FER()

def process_frame(frame):
    emotions = emotion_detector.detect_emotions(frame)
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
    
    for face in emotions:
        (x, y, w, h) = face["box"]
        emotion, score = max(face["emotions"].items(), key=lambda item: item[1])

        overlay = frame.copy()
        alpha = 0.4
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        blurred_frame[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        text = f"{emotion}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x + 10
        text_y = y - 10 if y - 10 > text_height else y + h + text_height

        cv2.rectangle(blurred_frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), bg_color, -1)
        cv2.putText(blurred_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return blurred_frame

st.title('Real-Time Emotion Recognition')
st.text('Open your webcam to start detecting emotions.')

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write('Error: Could not read frame.')
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = process_frame(frame)
    FRAME_WINDOW.image(processed_frame)

cap.release()
