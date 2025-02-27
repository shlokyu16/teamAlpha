import streamlit as st
import cv2
import numpy as np
from numpy.linalg import norm
import joblib
from tensorflow.keras.preprocessing.image import img_to_array
import PyPDF2
from PIL import Image
import tempfile

# Load pre-trained emotion detection model
model = joblib.load("emotion_model.pkl")
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
def get_expected_distribution(expected_emotion):
    """
    Defines a soft label distribution for expected emotions.
    This allows flexibility instead of strict one-hot matching.
    """
    emotion_map = {
        "Happy": {"Happy": 1.0, "Neutral": 0.3, "Surprise": 0.2},
        "Sad": {"Sad": 1.0, "Neutral": 0.5, "Fear": 0.2},
        "Angry": {"Angry": 1.0, "Disgust": 0.6, "Sad": 0.3},
        "Distressing": {"Fear": 1.0, "Surprise": 0.7, "Sad": 0.4},
        "Surprise": {"Surprise": 1.0, "Fear": 0.5, "Happy": 0.3},
        "Neutral": {"Neutral": 1.0, "Sad": 0.3, "Happy": 0.3},
        "Disgust": {"Disgust": 1.0, "Angry": 0.7, "Sad": 0.2}
    }
    return emotion_map.get(expected_emotion, {})

def hybrid_empathy_check(expected_emotion, predictions, prev_predictions=None):
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_conf = sorted_preds[0]
    second_emotion, second_conf = sorted_preds[1]

    # Confidence Check
    if expected_emotion == top_emotion and top_conf >= 0.95:
        confidence_score = top_conf
    elif expected_emotion == second_emotion and (top_conf - second_conf) < 0.15:
        confidence_score = second_conf * 0.9  # Slightly reduced weight
    else:
        confidence_score = 0

    # Multi-Emotion Similarity Mapping
    expected_dist = get_expected_distribution(expected_emotion)  # Get expected distribution
    detected_vec = np.array([predictions.get(em, 0) for em in predictions.keys()])
    expected_vec = np.array([expected_dist.get(em, 0) for em in predictions.keys()])
    similarity_score = cosine_similarity(expected_vec, detected_vec)

    # Context-Based Shift (if previous predictions exist)
    if prev_predictions:
        prev_score = prev_predictions.get(expected_emotion, 0)
        change_score = abs(predictions[expected_emotion] - prev_score)
    else:
        change_score = 0.5  # Neutral starting value

    # Final Score (weighted sum)
    final_score = (0.5 * confidence_score) + (0.3 * similarity_score) + (0.2 * change_score)

    return final_score

tabp = st.sidebar.radio("Choose tab: ", ["Empathy Check", "Model Confidence"])

if tabp == "Model Confidence":
    st.title("Model")
    st.write("Checks the models confidence in prediciting the emotion of a given face.")
    st.write("")
    st.write("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), maxSize=(1000, 1000))
        
        if len(faces) == 0:
            st.warning("No face detected!")
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_rgb = np.stack((roi_gray,) * 3, axis=-1)
                roi = roi_rgb.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)
                
                preds = model.predict(roi)
                confidence = np.max(preds)
                emotion = EMOTIONS[np.argmax(preds)]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
            st.info(f"Emotion: {emotion} (Confidence: {confidence:.2%})")

if tabp == "Empathy Check":
    st.title("Empathy Check System")
    st.write("This app analyzes facial expressions to assess empathy towards displayed content.")

    # Tabs for different content types
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Upload Video", "Read Content"])
    uploaded_content = None
    expected_emotion = ""

    with tab1:
        st.header("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        uploaded_content = uploaded_image

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with tab2:
        st.header("Upload a Video")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        uploaded_content = uploaded_video

        if uploaded_video is not None:
            st.video(uploaded_video)

    with tab3:
        st.header("Read Content")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        displayed_text = ""
        
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                displayed_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                displayed_text = "Error"
                
        uploaded_content = uploaded_file
        st.text_area("Content:", displayed_text, height=500)

    # Capture video feed
    cap = cv2.VideoCapture(0)
    emotion = ""

    # Expanded expected emotions dictionary with grouped emotions
    expected_emotions = [
        "Default",
        "Sad",
        "Happy",
        "Distressing",
        "Neutral",
        "Angry",
        "Disgusted",
        "Surprised"
    ]

    content_type = st.radio("Select Content Type", list(expected_emotions))
    if (content_type != "default"):
        expected_emotion = content_type
        
    p_preds = None

    emotionp = st.empty()
    empathyp = st.empty()

    if uploaded_content is not None:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_rgb = np.stack((roi_gray,) * 3, axis=-1)
                    roi = roi_rgb.astype("float") / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi)[0]
                    confidence = np.max(preds)
                    emotion = EMOTIONS[np.argmax(preds)]
                    preds = {EMOTIONS[i]: preds[i] for i in range(len(EMOTIONS))}

                    emotionp.write(f"Emotion: {emotion} (Confidence: {confidence:.2%})")

                    empathy_score = hybrid_empathy_check(expected_emotion, preds, p_preds)
                    p_preds = preds
                    
                    if empathy_score >= 0.35:
                        empathyp.success("Empathy detected!")
                    else:
                        empathyp.warning("No Empathy detected.")
