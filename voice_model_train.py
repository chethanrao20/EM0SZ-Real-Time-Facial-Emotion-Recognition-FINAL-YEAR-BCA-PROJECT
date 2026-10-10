# voice_model_train.py
# Extract MFCC features and train a simple SVM classifier.
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

DATA_DIR = 'datasets/RAVDESS'  # expected folder structure: DATA_DIR/<emotion>/*.wav

def extract_features(path, sr=22050, duration=3):
    try:
        y, sr = librosa.load(path, sr=sr, duration=duration, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print('Error', path, e)
        return None

def main():
    X, y = [], []
    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            if not file.lower().endswith('.wav'): continue
            fp = os.path.join(folder, file)
            feat = extract_features(fp)
            if feat is None: continue
            X.append(feat)
            y.append(label)
    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    print('Train accuracy:', clf.score(X_train, y_train))
    print('Test accuracy:', clf.score(X_test, y_test))
    joblib.dump((clf, le), 'voice_emotion_svm.joblib')

if __name__ == '__main__':
    main()
