# utils.py - helper functions
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def load_face_model(path='face_emotion_model.h5'):
    return tf.keras.models.load_model(path)

def load_voice_model(path='voice_emotion_svm.joblib'):
    clf, le = joblib.load(path)
    return clf, le

def preprocess_face(gray_face):
    roi = cv2.resize(gray_face, (48,48))
    roi = roi.astype('float')/255.0
    roi = roi.reshape(1,48,48,1)
    return roi

