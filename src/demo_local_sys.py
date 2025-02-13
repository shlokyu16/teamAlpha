import streamlit as st
import cv2
import numpy as np
import joblib
import dlib
from scipy.spatial import distance
from collections import deque
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Emotion labels based on FER2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DROWSY_LABELS = ['Sad', 'Neutral', 'Happy', 'Surprise']  # More weight on these emotions

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    # Compute the eye aspect ratio (EAR) to detect blinking/drowsiness.
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Eye landmark indexes (from dlib's 68-point model)
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Track past emotions & EAR scores
history_length = 30  # Frames to track
if "emotion_history" not in st.session_state or st.session_state.emotion_history is None:
    st.session_state.emotion_history = deque(maxlen=history_length)
if "ear_history" not in st.session_state or st.session_state.ear_history is None:
    st.session_state.ear_history = deque(maxlen=history_length)
EAR_THRESHOLD = 0.1

# Gaze threshold (relative to eye position in frame)
GAZE_THRESHOLD = 0.15  # If the iris shifts too far from the center, it's considered distracted

def detect_gaze(landmarks, frame):
    # Determine if the driver is looking away from the camera using eye landmarks.
    left_eye_center = np.mean(landmarks[LEFT_EYE], axis=0)
    right_eye_center = np.mean(landmarks[RIGHT_EYE], axis=0)

    # Calculate the horizontal offset of the eyes from the frame center
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    left_offset = left_eye_center[0] - frame_center[0]
    right_offset = right_eye_center[0] - frame_center[0]

    # Average offset of both eyes
    avg_offset = (abs(left_offset) + abs(right_offset)) / 2
    gaze_ratio = avg_offset / (frame.shape[1] / 2)

    return gaze_ratio > GAZE_THRESHOLD  # Return True if the gaze is off-center

# Streamlit UI
st.title("Driver Alertness Detector")
tabs = ["Live Detection", "Upload Image", "Upload Video"]
choice = st.sidebar.radio("Choose Mode:", tabs)

if choice == "Live Detection":

    # Load the trained model
    MODEL_PATH = "emotion_model.h5"
    model = load_model(MODEL_PATH)
    
    st.write("Detects fatigue based on facial expressions and eye movement")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    gaze_off_center = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            if gaze_off_center:
                alert_placeholder.warning("⚠️ Low alertness detected! Stay focused!")
            else:
                alert_placeholder.info("😶 Face not detected! Stay focused!")
            continue

        for face in faces:
            shape = predictor(gray, face)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # Extract eyes and compute EAR
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            ear = (left_EAR + right_EAR) / 2.0
            if len(st.session_state.ear_history) < history_length:
                st.session_state.ear_history.append(ear)

            # Emotion detection
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (48, 48))  # FER2013 uses 48x48
            roi = img_to_array(roi)
            roi = np.stack([roi] * 3, axis=-1)  # Convert grayscale to 3-channel (RGB)
            roi = roi.reshape(1, 48, 48, 3) / 255.0  # Normalize

            preds = model.predict(roi)
            emotion_idx = np.argmax(preds)
            emotion = EMOTIONS[emotion_idx]
            if len(st.session_state.emotion_history) < history_length:
                st.session_state.emotion_history.append(emotion)

            # Compute alertness
            drowsy_emotions = sum(1 for e in st.session_state.emotion_history if e in DROWSY_LABELS)
            avg_ear = np.mean(st.session_state.ear_history) if st.session_state.ear_history else 1.0
            gaze_off_center = detect_gaze(landmarks, frame)

            if drowsy_emotions > (history_length * 0.6) and (avg_ear < EAR_THRESHOLD or gaze_off_center):
                alert_placeholder.warning("⚠️ Low alertness detected! Stay focused!")
                color = (0, 0, 255)
            else:
                alert_placeholder.success("✅ Focused!")
                color = (0, 255, 0)

            # Draw rectangle and labels
            cv2.rectangle(frame, (x, y), (x + w + 5, y + h +5), color, 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show video in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

elif choice == "Upload Image":

    # Load the trained model
    MODEL_PATH = "emotion_model.h5"
    model = load_model(MODEL_PATH)

    st.write("Upload an image for emotion detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "pdf", ".heic"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) == 0:
            st.warning("No face detected!")
        else:
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = img_to_array(roi)
                roi = np.stack([roi] * 3, axis=-1)
                roi = roi.reshape(1, 48, 48, 3) / 255.0
                
                preds = model.predict(roi)
                emotion = EMOTIONS[np.argmax(preds)]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
            st.info(f"Emotion: {emotion}")

elif choice == "Upload Video":

    # Load the trained model
    MODEL_PATH = "emotion_model.h5"
    model = load_model(MODEL_PATH)

    st.write("Upload a video for alertness analysis")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.write("Saving ...")
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.write("Analyzing ...")
        cap = cv2.VideoCapture(video_path)
        frame_count, drowsy_frames = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = predictor(gray, face)
                landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                ear = (eye_aspect_ratio(landmarks[LEFT_EYE]) + eye_aspect_ratio(landmarks[RIGHT_EYE])) / 2.0
                
                # Emotion detection
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (48, 48))  # FER2013 uses 48x48
                roi = img_to_array(roi)
                roi = np.stack([roi] * 3, axis=-1)  # Convert grayscale to 3-channel (RGB)
                roi = roi.reshape(1, 48, 48, 3) / 255.0  # Normalize

                preds = model.predict(roi)
                emotion_idx = np.argmax(preds)
                emotion = EMOTIONS[emotion_idx]
                
                if emotion in DROWSY_LABELS and ear < EAR_THRESHOLD:
                    drowsy_frames += 1
                frame_count += 1
        cap.release()
        
        alertness_score = 100 - (drowsy_frames / frame_count * 100)
        if (alertness_score < 65.0):
            st.warning(f"Alertness Score: {alertness_score:.2f}%")
        elif (alertness_score > 95.0):
            st.success(f"Alertness Score: {alertness_score:.2f}%")
        else:
            st.info(f"Alertness Score: {alertness_score:.2f}%")
