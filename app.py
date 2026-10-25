
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="FaceAI | Face Detection",
    page_icon="📸",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(120deg, #182848, #315b96);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #e0e9ff;
    font-size: 17px;
}
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📸 FaceAI")
    st.caption("Computer Vision Project")
    st.divider()
    st.subheader("About")
    st.write(
        "Detect human faces in images using "
        "Python, OpenCV, and Haar Cascade."
    )
    st.info("Upload a JPG, JPEG, or PNG image to get started.")
    st.caption("Version 1.0 | Built with Streamlit")

st.markdown("""
<div class="hero">
    <h1>Face Detection System</h1>
    <p>Detect, count, and highlight faces in your images.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Upload an image")
uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result = original.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 220, 100),
            3
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(result, use_column_width=True)

    st.divider()
    st.subheader("Analysis Summary")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Faces Detected", len(faces))

    with m2:
        st.metric("Image Width", f"{image.width} px")

    with m3:
        st.metric("Image Height", f"{image.height} px")

    st.divider()

    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )

    if success:
        st.download_button(
            "⬇️ Download Detection Result",
            data=encoded.tobytes(),
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )

    if len(faces) == 0:
        st.warning(
            "No faces detected. Try a clear, front-facing photo "
            "with good lighting."
        )
    else:
        st.success(
            f"Analysis complete! Detected {len(faces)} face(s)."
        )

else:
    st.info(
        "👆 Upload an image above to see the original image, "
        "detection results, and analysis metrics."
    )

st.divider()
st.caption(
    "FaceAI | Python • OpenCV • NumPy • Streamlit"
)
