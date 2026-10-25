
import hashlib
import re
import html

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from autonomous_agent import run_autonomous_agent


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FaceAI Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# REPORT FORMATTING
# --------------------------------------------------

def clean_ai_report(report):
    """Convert common HTML formatting into readable Markdown."""
    if not isinstance(report, str):
        return "The AI agent did not return a valid text report."

    # Convert HTML line breaks into newlines.
    report = re.sub(r"<br\s*/?>", "\n", report, flags=re.IGNORECASE)

    # Convert HTML paragraph endings into line breaks.
    report = re.sub(r"</p\s*>", "\n\n", report, flags=re.IGNORECASE)
    report = re.sub(r"<p(?:\s[^>]*)?>", "", report, flags=re.IGNORECASE)

    # Decode common HTML entities such as &amp; and &nbsp;.
    report = html.unescape(report)

    # Remove any remaining HTML tags so they are not displayed as raw code.
    report = re.sub(r"<[^>]+>", "", report)

    # Normalize excessive blank lines.
    report = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", report)

    return report.strip()


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
/* Main background */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #0b1120;
    color: #e2e8f0;
}

[data-testid="stHeader"] {
    background-color: #0b1120;
}

/* Main text */
.stApp p,
.stApp li,
.stApp label,
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown td,
.stMarkdown th {
    color: #e2e8f0;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #f8fafc !important;
}

/* Main title */
.main-title {
    color: #f8fafc !important;
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    color: #cbd5e1 !important;
    font-size: 1.08rem;
    margin-bottom: 18px;
}

/* Section titles */
.section-title {
    color: #f8fafc !important;
    font-size: 1.45rem;
    font-weight: 750;
    margin: 10px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #e5e7eb !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #172033;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #ffffff !important;
}

[data-testid="stMetricDelta"] {
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #60a5fa !important;
    border-radius: 10px;
    font-weight: 650;
    min-height: 42px;
}

.stButton > button *,
.stDownloadButton > button * {
    color: #ffffff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #93c5fd !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #172033;
    border: 1px dashed #64748b;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] * {
    color: #f1f5f9 !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-color: #475569 !important;
}

/* Select controls */
[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #172033;
    border: 1px solid #334155;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li {
    color: #f1f5f9 !important;
}

/* Markdown report tables */
.stMarkdown table {
    width: 100%;
    border-collapse: collapse;
    color: #e2e8f0 !important;
}

.stMarkdown thead tr th {
    background-color: #263449 !important;
    color: #ffffff !important;
    padding: 10px;
    border: 1px solid #475569 !important;
}

.stMarkdown tbody tr td {
    background-color: #172033 !important;
    color: #e2e8f0 !important;
    padding: 10px;
    border: 1px solid #334155 !important;
}

/* Code blocks */
[data-testid="stCode"] {
    background-color: #172033;
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Captions */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #cbd5e1 !important;
}

/* Links */
.stApp a {
    color: #93c5fd !important;
}

/* Dividers */
hr {
    border-color: #334155 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "image_signature" not in st.session_state:
    st.session_state.image_signature = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🤖 FaceAI Pro")

    st.caption("Autonomous Computer Vision Assistant")

    st.divider()

    st.subheader("Features")

    st.markdown("""
    - Face detection
    - Face bounding boxes
    - Image sharpness analysis
    - Brightness analysis
    - AI-generated reports
    - Downloadable results
    """)

    st.divider()

    st.subheader("AI Model")
    st.write("NVIDIA Nemotron 3 Ultra 550B")

    st.caption(
        "The AI agent selects analysis tools and "
        "uses their results to prepare a report."
    )

    st.divider()

    st.caption("Computer vision: OpenCV + NumPy")
    st.caption("Interface: Streamlit")


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 FaceAI Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered face detection and image quality analysis'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload an image and run the autonomous agent. "
    "The agent selects available analysis tools, executes "
    "them locally, and generates a report."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Choose a JPG or PNG image."
)

if uploaded_file is None:
    st.session_state.analysis_result = None
    st.session_state.image_signature = None

    st.markdown("### Get started")

    st.write(
        "Upload an image to begin your analysis. "
        "Image processing uses local computer vision functions. "
        "Analysis results are sent to the configured NVIDIA model "
        "to generate the report."
    )

    st.stop()


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image_bytes = uploaded_file.getvalue()
current_signature = hashlib.sha256(image_bytes).hexdigest()

if st.session_state.image_signature != current_signature:
    st.session_state.analysis_result = None
    st.session_state.image_signature = current_signature

try:
    pil_image = Image.open(uploaded_file)
    image_format = pil_image.format or "Image"
    pil_image = pil_image.convert("RGB")
    original = np.array(pil_image)

except Exception as error:
    st.error(f"Could not read the uploaded image: {error}")
    st.stop()


# --------------------------------------------------
# DISPLAY UPLOADED IMAGE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Uploaded Image</div>',
    unsafe_allow_html=True
)

image_col, details_col = st.columns([1.2, 1])

with image_col:
    st.image(
        original,
        caption="Original image",
        use_column_width=True
    )

with details_col:
    st.markdown("### Image information")

    height, width = original.shape[:2]

    st.metric("Image width", f"{width} px")
    st.metric("Image height", f"{height} px")

    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**Format:** {image_format}")
    st.write(f"**Dimensions:** {width} × {height} pixels")


# --------------------------------------------------
# RUN AUTONOMOUS AGENT
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Autonomous AI Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "The agent can select available analysis tools, "
    "execute them, inspect their results, and prepare a report."
)

run_button = st.button(
    "🚀 Run Autonomous Analysis",
    type="primary",
    key="run_autonomous_analysis"
)

if run_button:
    try:
        with st.spinner(
            "The AI agent is selecting tools and analyzing "
            "your image. This may take a little time..."
        ):
            result = run_autonomous_agent(original)

        st.session_state.analysis_result = result

    except Exception as error:
        st.session_state.analysis_result = None

        st.error(
            "The autonomous analysis failed. Check your NVIDIA "
            "API connection and the terminal for more details."
        )
        st.code(str(error))


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.analysis_result

if result is None:
    st.info("Click Run Autonomous Analysis when you're ready.")
    st.stop()


# --------------------------------------------------
# AGENT ACTIVITY
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Agent Activity</div>',
    unsafe_allow_html=True
)

steps = result.get("steps", [])
tools_executed = result.get("tool_calls", [])

if steps:
    for step in steps:
        st.success(str(step))
else:
    st.warning("No analysis steps were completed.")

if tools_executed:
    st.write("**Tools selected and executed:**")

    for tool_name in tools_executed:
        st.code(str(tool_name), language="text")
else:
    st.warning("The agent did not execute any tools.")


# --------------------------------------------------
# FACE DETECTION RESULTS
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Face Detection Results</div>',
    unsafe_allow_html=True
)

detection = result.get("detection")
quality = result.get("quality")

if detection is not None:
    face_count = detection.get("face_count", 0)
    bounding_boxes = detection.get("bounding_boxes", [])

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Faces detected", face_count)

    with metric_col2:
        st.metric("Bounding boxes", len(bounding_boxes))

    annotated_image = detection.get("annotated_image")

    if annotated_image is not None:
        st.image(
            annotated_image,
            caption="Image with detected faces",
            use_column_width=True
        )

        success, encoded_image = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_RGB2BGR
            )
        )

        if success:
            st.download_button(
                label="Download Annotated Image",
                data=encoded_image.tobytes(),
                file_name="faceai_detected_faces.jpg",
                mime="image/jpeg",
                key="download_annotated_image"
            )

    if bounding_boxes:
        st.markdown("#### Face locations")

        st.dataframe(
            bounding_boxes,
            use_container_width=True
        )
    else:
        st.info("No faces were detected in this image.")

else:
    st.warning(
        "Face detection was not completed. "
        "The agent may have stopped before running this tool."
    )


# --------------------------------------------------
# IMAGE QUALITY RESULTS
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Image Quality Analysis</div>',
    unsafe_allow_html=True
)

if quality is not None:
    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric(
            "Sharpness score",
            quality.get("sharpness_score", "N/A")
        )
        st.caption(
            quality.get("sharpness_assessment", "Not available")
        )

    with quality_col2:
        st.metric(
            "Average brightness",
            quality.get("average_brightness", "N/A")
        )
        st.caption(
            quality.get("brightness_assessment", "Not available")
        )

    with quality_col3:
        image_width = quality.get("width", "N/A")
        image_height = quality.get("height", "N/A")

        st.metric(
            "Resolution",
            f"{image_width} × {image_height}"
        )

    st.caption(
        "Sharpness and brightness are estimates based on pixel "
        "statistics. They are not absolute measurements."
    )

else:
    st.warning("Image quality analysis was not completed.")


# --------------------------------------------------
# AI-GENERATED REPORT
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">AI-Generated Report</div>',
    unsafe_allow_html=True
)

raw_report = result.get(
    "report",
    "The agent did not return a report."
)

report = clean_ai_report(raw_report)

if report:
    st.markdown(report)

    st.download_button(
        label="Download AI Report",
        data=report,
        file_name="faceai_autonomous_report.md",
        mime="text/markdown",
        key="download_ai_report"
    )

else:
    st.warning("The agent returned an empty report.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FaceAI Pro | NVIDIA Nemotron | OpenCV | NumPy | Streamlit"
)

st.caption(
    "This application analyzes image pixels and does not "
    "identify people or verify their identities."
)
