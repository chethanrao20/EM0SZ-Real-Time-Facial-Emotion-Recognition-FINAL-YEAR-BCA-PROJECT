import streamlit as st
import cv2
import av
import threading
import time

from deepface import DeepFace
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)

st.set_page_config(
    page_title="EmotionSense",
    page_icon=None,
    layout="wide"
)

st.title("EmotionSense")
st.subheader("Real-Time Facial Emotion Recognition")

st.write(
    "This application analyzes webcam frames using DeepFace "
    "and a pretrained facial emotion recognition model."
)

st.warning(
    "Emotion predictions are estimates from facial appearance, "
    "not reliable measurements of a person's internal feelings."
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.last_result = "Waiting for a face"
        self.last_analysis = 0.0
        self.analysis_interval = 2.0
        self.busy = False

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        detector = cv2.CascadeClassifier(cascade_path)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        now = time.time()

        if len(faces) > 0:
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 200, 100),
                2
            )

            with self.lock:
                can_analyze = (
                    not self.busy
                    and now - self.last_analysis
                    >= self.analysis_interval
                )

                if can_analyze:
                    self.busy = True
                    self.last_analysis = now

            if can_analyze:
                try:
                    face_crop = image[y:y + h, x:x + w].copy()

                    result = DeepFace.analyze(
                        img_path=face_crop,
                        actions=["emotion"],
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True
                    )

                    if result:
                        emotion_data = result[0]["emotion"]
                        emotion = max(
                            emotion_data,
                            key=emotion_data.get
                        )
                        confidence = emotion_data[emotion]

                        message = (
                            f"{emotion.title()} "
                            f"({confidence:.1f}%)"
                        )
                    else:
                        message = "No result returned"

                    with self.lock:
                        self.last_result = message

                except Exception as error:
                    with self.lock:
                        self.last_result = (
                            "Analysis error: "
                            + str(error)[:120]
                        )
                finally:
                    with self.lock:
                        self.busy = False

            with self.lock:
                label = self.last_result

            cv2.putText(
                image,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 100),
                2
            )

        else:
            with self.lock:
                self.last_result = "Waiting for a face"

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


st.markdown("### Live Camera")

st.write(
    "Start the camera and face it directly toward your webcam. "
    "Analysis runs periodically to reduce processing load."
)

webrtc_streamer(
    key="emotionsense-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.caption(
    "Processing is performed locally by this application. "
    "The first analysis may take longer while model weights are downloaded."
)
