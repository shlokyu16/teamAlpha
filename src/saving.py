import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("emotion_model.pkl", compile=False) 
model.save("emotion_model_tf")  
