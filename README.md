# 📸 FaceAI — Face Detection System

A computer vision application built using Python, OpenCV, and Streamlit to detect and count human faces in images. The application highlights detected faces with bounding boxes and allows users to download the processed image.

## ✨ Features

- **Image Face Detection:** Detect faces in JPG, JPEG, and PNG images.
- **Multiple Face Detection:** Detect multiple faces in a single image.
- **Bounding Boxes:** Highlight detected faces with green rectangles.
- **Face Counting:** Display the total number of detected faces.
- **Image Analysis:** Display the image dimensions and detection results.
- **Download Results:** Download the processed image with bounding boxes.
- **Live Webcam Detection:** Run real-time face detection using a webcam through the standalone webcam script.
- **Interactive Web Interface:** Upload and analyze images through a Streamlit application.

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit
- Haar Cascade Classifier

## 📂 Project Structure

```text
face-detection-system/
├── app.py
├── detect.py
├── webcam.py
├── requirements.txt
├── README.md
├── person.jpg
└── screenshots/
```

`person.jpg` is an optional local test image. The `screenshots/` folder can contain screenshots of your application.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/face-detection-system.git
cd face-detection-system
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL you create.

### 2. Create a virtual environment

For Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 🚀 How to Run

### Run the Streamlit web application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Upload an image to view the original image, detected faces, face count, and download option.

### Run image detection

Place a test image named `person.jpg` in the project directory, then run:

```bash
python detect.py
```

### Run webcam detection

```bash
python webcam.py
```

Press the **ESC** key to close the webcam window.

## 🧠 How It Works

1. Load the input image or capture a frame from the webcam.
2. Convert the image to grayscale.
3. Load OpenCV's pre-trained Haar Cascade frontal-face classifier.
4. Detect face-like regions using `detectMultiScale()`.
5. Draw bounding boxes around detected faces.
6. Count the detected faces and display the results.

## 📌 Face Detection vs. Face Recognition

**Face detection** determines whether faces are present in an image and identifies their locations.

**Face recognition** attempts to determine whose face it is.

This project implements face detection, not identity recognition.

## ⚠️ Limitations

- Detection accuracy depends on lighting, image quality, face angle, and visibility.
- Haar Cascade may miss faces or produce false detections.
- The frontal-face model is less reliable for profiles and heavily obstructed faces.
- The application does not identify individuals.

## 🔮 Future Improvements

- Add face cropping and face image export.
- Support video file uploads.
- Improve detection using a modern deep-learning-based face detector.
- Add detection confidence information where supported by the selected model.
- Deploy the application online.

## 👨‍💻 Author

**Jaswanth Putta**

GitHub: https://github.com/jaswanthputta

## 📄 License

This project is available for learning and educational purposes. Add an explicit open-source license if you want to define permissions for reuse and redistribution.
