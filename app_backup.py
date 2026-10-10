# app.py - Streamlit interface for EmotionSense (image & audio upload)
import streamlit as st
from utils import EMOTION_LABELS, preprocess_face, load_face_model, load_voice_model
import cv2
import numpy as np
import tempfile
import librosa
import joblib

st.title('EmotionSense Demo')

face_model = None
voice_model = None

if st.button('Load Models'):
    face_model = load_face_model('face_emotion_model.h5')
    voice_clf, voice_le = load_voice_model('voice_emotion_svm.joblib')
    st.success('Models loaded (ensure files exist in working dir)')

st.header('Image (Face) Prediction')
img = st.file_uploader('Upload face image', type=['png','jpg','jpeg'])
if img is not None:
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    from utils import preprocess_face
    roi = preprocess_face(gray)
    if face_model is None:
        face_model = load_face_model('face_emotion_model.h5')
    preds = face_model.predict(roi)
    st.write('Predicted emotion:', EMOTION_LABELS[np.argmax(preds)])

st.header('Audio (Voice) Prediction')
audio = st.file_uploader('Upload .wav audio (short clip)', type=['wav'])
if audio is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tfile.write(audio.read())
    tfile.flush()
    y, sr = librosa.load(tfile.name, sr=22050, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = np.mean(mfcc.T, axis=0).reshape(1,-1)
    if voice_model is None:
        voice_clf, voice_le = load_voice_model('voice_emotion_svm.joblib')
    pred = voice_clf.predict(feat)
    label = voice_le.inverse_transform(pred)[0]
    st.write('Predicted voice emotion:', label)

