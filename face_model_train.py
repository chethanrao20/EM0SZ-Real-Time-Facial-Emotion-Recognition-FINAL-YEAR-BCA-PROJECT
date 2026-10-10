# face_model_train.py
# Minimal training scaffold for FER2013 using Keras.
import os
import numpy as np
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# NOTE: This is scaffold code. Replace data loading with actual FER2013 parsing.
IMG_SIZE = 48
BATCH_SIZE = 64
NUM_CLASSES = 7

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256,(3,3),activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128,activation='relu'),
        layers.Dense(NUM_CLASSES,activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # TODO: load datasets from datasets/FER2013 (CSV or images)
    # Using ImageDataGenerator assumes folder structure: datasets/FER2013/train/<label>/*.png
    train_dir = 'datasets/FER2013/train'
    val_dir = 'datasets/FER2013/val'
    datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_gen = datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE,IMG_SIZE),
                                            color_mode='grayscale', batch_size=BATCH_SIZE,
                                            class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(IMG_SIZE,IMG_SIZE),
                                          color_mode='grayscale', batch_size=BATCH_SIZE,
                                          class_mode='categorical')
    model = build_model()
    checkpoint = ModelCheckpoint('face_emotion_model.h5', save_best_only=True, monitor='val_accuracy')
    model.fit(train_gen, validation_data=val_gen, epochs=25, callbacks=[checkpoint])

if __name__ == '__main__':
    main()
