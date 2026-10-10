# EmotionSense: Real-Time Emotion Detection using Face and Voice

## Abstract
EmotionSense is a multimodal system that detects human emotions using facial expressions and voice tone. The system fuses predictions from a CNN-based facial model and an SVM-based audio model to improve robustness in real-world conditions.

## Introduction
(Include background, motivation, and applications: online learning, mental health, HCI.)

## Literature Review
(Short review: FER2013 studies, audio-emotion RAVDESS benchmarks, multimodal fusion techniques.)

## Methodology
- Face pipeline: detect face (Haar cascade or MTCNN), preprocess to 48x48 grayscale, CNN classifier with conv + pool layers.
- Audio pipeline: load audio, extract MFCCs (n_mfcc=40), average frames → feature vector, SVM or RandomForest classifier.
- Fusion: weighted average of probabilities; simple rule-based fallback.

## Implementation
(Describe training, hyperparameters, and evaluation.)

## Results
(Insert accuracy, confusion matrices, precision/recall. Add screenshots in final PDF.)

## Conclusion & Future Work
- Improve fusion (attention-based), use transformer audio models, deploy on mobile.

## References
- FER2013, RAVDESS datasets; relevant papers and tutorials.

