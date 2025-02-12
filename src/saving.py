import joblib
from tensorflow.keras.models import load_model

model = joblib.load("emotion_model.pkl")
model.save("emotion_model.h5")
