# realtime_demo.py
# Simple webcam demo that shows face emotion predictions.
import cv2
from utils import load_face_model, preprocess_face, EMOTION_LABELS
import numpy as np

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = load_face_model('face_emotion_model.h5')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            inp = preprocess_face(roi)
            preds = model.predict(inp)
            label = EMOTION_LABELS[np.argmax(preds)]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('EmotionSense - Face Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
