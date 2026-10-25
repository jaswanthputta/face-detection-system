# FaceAI Pro — Autonomous Computer Vision Assistant

FaceAI Pro is an AI-powered computer vision application built with Python, OpenCV, NumPy, Streamlit, and NVIDIA Nemotron. It detects faces in uploaded images, evaluates image quality, and generates an AI-written analysis report using an autonomous tool-calling workflow.

## Features

- **Face Detection:** Detects faces in uploaded JPG and PNG images using OpenCV's Haar Cascade classifier.
- **Face Localization:** Draws bounding boxes around detected faces and displays their coordinates.
- **Image Quality Analysis:** Estimates image sharpness and average brightness.
- **Autonomous AI Agent:** Selects available analysis tools, executes them, and uses their results to prepare a report.
- **NVIDIA AI Integration:** Uses the NVIDIA API with the Nemotron 3 Ultra model for AI-generated reports.
- **Interactive Web Interface:** Provides an easy-to-use Streamlit interface for uploading and analyzing images.
- **Downloadable Results:** Download annotated images and AI-generated reports.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and agent logic |
| OpenCV | Face detection and image processing |
| NumPy | Image data and numerical calculations |
| Streamlit | Interactive web interface |
| NVIDIA API | AI model integration |
| NVIDIA Nemotron 3 Ultra | AI reasoning and report generation |
| Requests | API communication |
| python-dotenv | Loading environment variables |
| Pillow | Image loading and conversion |

## How It Works

1. The user uploads a JPG or PNG image through the Streamlit interface.
2. The autonomous agent receives the image and the available tool definitions.
3. The AI agent selects image-analysis tools, such as face detection and image quality analysis.
4. The selected Python functions process the image locally.
5. The results are returned to the agent for report generation.
6. The interface displays detected faces, bounding boxes, image-quality metrics, agent activity, and the AI report.
7. The user can download the annotated image and report.

## Project Structure

```text
face-detection-system/
│
├── app.py                  # Streamlit application
├── autonomous_agent.py     # Autonomous AI agent and tool execution
├── analysis_tools.py       # Face detection and image quality functions
├── ai_agent.py             # AI report generation workflow
├── agent_workflow.py       # Fixed analysis workflow
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Files excluded from Git
├── .env                    # Private API configuration (not committed)
│
└── person.jpg              # Optional local test image
```

## Prerequisites

- Python 3.10 or later
- pip
- A valid NVIDIA API key with access to the configured model
- Internet connectivity for NVIDIA API requests

The project has been tested with Python 3.12.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jaswanthputta/face-detection-system.git
cd face-detection-system
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the NVIDIA API key

Create a `.env` file in the project root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Replace the placeholder with your own API key. Never publish your real API key or commit the `.env` file to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser to use FaceAI Pro.

## Usage

1. Open FaceAI Pro in your browser.
2. Upload a JPG or PNG image.
3. Click **Run Autonomous Analysis**.
4. Review the agent's activity and executed tools.
5. View the detected faces and bounding-box coordinates.
6. Review the sharpness, brightness, and resolution metrics.
7. Read the generated AI report.
8. Download the annotated image or report if needed.

## Image Quality Metrics

The application calculates:

- **Sharpness score:** Uses the variance of the Laplacian to estimate image sharpness.
- **Average brightness:** Uses grayscale pixel values to estimate brightness.
- **Resolution:** Displays the image width and height in pixels.

These metrics are heuristic estimates. Their interpretation can vary with image content, lighting, and resolution.

## Security and Privacy

- Keep API keys in environment variables rather than source code.
- Keep `.env` and other private configuration files out of Git commits.
- Use images that you have permission to process.
- Image processing functions run locally, while the AI report workflow sends analysis information to the configured NVIDIA API.
- Do not assume uploaded image data is never transmitted externally; review the agent implementation and API provider's policies before using sensitive images.

## Limitations

- Haar Cascade face detection may miss faces or produce false detections in difficult lighting, unusual angles, or low-quality images.
- Sharpness and brightness scores do not guarantee that an image is suitable for every purpose.
- AI-generated reports may contain errors and should be treated as suggestions.
- NVIDIA API availability, model access, and response times depend on the configured service and account.

## Future Improvements

- Add webcam-based face detection.
- Improve detection using a modern deep-learning model.
- Add more image-quality metrics.
- Export structured analysis results as JSON or CSV.
- Add automated tests and more robust error handling.
- Improve deployment and configuration documentation.

## Author

**Jaswanth Putta**

GitHub: [@jaswanthputta](https://github.com/jaswanthputta)

## License

Add a license file if you intend to distribute this project for reuse. Until a license is specified, do not assume others have permission to reuse the code.
