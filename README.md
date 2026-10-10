# EM0SZ — Real-Time Facial Emotion Recognition

**Read the face. Decode the emotion.**

EM0SZ is a real-time facial emotion recognition application developed as a BCA academic project. It uses a webcam to detect a person's face and estimate the emotion expressed in the facial image.

The application provides a simple, interactive interface built with Streamlit and processes live webcam video using OpenCV, DeepFace, and Streamlit-WebRTC.

## About the Project

Facial expressions are an important part of human communication. EM0SZ demonstrates how computer vision and deep learning can be used to analyze facial expressions and estimate emotions from a live video feed.

The project is designed as a beginner-friendly demonstration of real-time facial emotion recognition without requiring users to train a machine learning model from scratch.

## Features

- **Real-Time Webcam Detection:** Processes live webcam video.
- **Face Detection:** Identifies faces in video frames.
- **Emotion Recognition:** Estimates facial emotions using DeepFace.
- **Interactive Web Interface:** Uses Streamlit for the application interface.
- **Live Video Processing:** Uses Streamlit-WebRTC to handle webcam streaming.
- **Python-Based Implementation:** Uses Python and computer vision libraries.
- **Academic Project:** Suitable for demonstrating basic concepts in AI, computer vision, and facial emotion recognition.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| OpenCV | Image processing and face detection |
| DeepFace | Facial emotion analysis |
| TensorFlow / Keras | Deep learning support |
| Streamlit-WebRTC | Real-time webcam streaming |
| PyAV | Video frame processing |
| Git and GitHub | Version control and project backup |

## How It Works

1. The user opens the EM0SZ application in a web browser.
2. The application requests permission to access the webcam.
3. The webcam captures live video frames.
4. OpenCV helps process the frames and detect faces.
5. DeepFace analyzes the detected face and estimates its emotion.
6. The application displays the processed video and emotion result through the web interface.

## Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.12
- pip
- A webcam
- A modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/em0sz.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the Project Folder

```bash
cd em0sz
```

### Step 3: Create a Virtual Environment

```bash
py -3.12 -m venv venv
```

### Step 4: Activate the Virtual Environment

On Windows Command Prompt:

```bat
venv\Scripts\activate
```

### Step 5: Install Dependencies

If the project includes a `requirements.txt` file, run:

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser and allow webcam access when prompted.

## Project Structure

```text
EM0SZ/
│
├── app.py                 # Main Streamlit application
├── utils.py               # Utility functions, if used
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── app_backup.py          # Backup of the application, if included
├── utils_backup.py        # Backup of utility code, if included
└── .gitignore              # Files excluded from Git
```

The files marked as optional should only appear in this structure if you have included them in your repository.

## Requirements

The project uses Python libraries for web development, computer vision, video streaming, and emotion analysis.

Install the dependencies from `requirements.txt` to reproduce the project environment. The required packages should match the versions used by the working application.

## Troubleshooting

**Webcam is not working**
- Allow camera permission in your browser.
- Close other applications that may be using the webcam.
- Check that your webcam is connected and working.

**Module not found**
- Activate the project's virtual environment.
- Install the dependencies using `python -m pip install -r requirements.txt`.

**Application does not start**
- Make sure you are running the command from the project directory.
- Confirm that Python 3.12 is installed.
- Review the error message in the terminal.

**First run takes longer**
- DeepFace may need to download or initialize model files on the first run. An internet connection may be required for initial setup.

## Limitations

- Emotion predictions may not always be accurate.
- Lighting, camera quality, facial angles, and image quality can affect the results.
- Facial expressions do not always represent a person's actual emotional state.
- Performance depends on the computer's hardware and installed dependencies.

## Privacy and Responsible Use

EM0SZ is intended for educational and demonstration purposes. Facial emotion predictions are estimates, not definitive conclusions about a person's feelings, personality, or mental state.

Use the application responsibly. Obtain permission before recording or analyzing another person's face, and avoid using predictions to make important decisions about individuals.

## Future Improvements

Possible future enhancements include:

- Improved user interface and theme customization.
- Better visualization of detected emotions.
- Emotion detection history during a session.
- Additional testing under different lighting conditions.
- Improved performance and usability.

## Author

**Chethan Rao**

BCA Academic Project  
**Project:** EM0SZ — Real-Time Facial Emotion Recognition

## License

This project is intended for educational purposes. If you plan to publish or redistribute it, add a license that reflects how you want others to use your code.

---

**EM0SZ — Read the face. Decode the emotion.**
