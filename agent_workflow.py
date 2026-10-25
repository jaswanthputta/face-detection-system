
from analysis_tools import detect_faces, analyze_image_quality
from ai_agent import generate_analysis_report


def run_face_analysis_agent(image_rgb):
    """
    Coordinate computer vision tools and generate
    a report using the NVIDIA-powered AI function.
    """

    steps = []

    # Step 1: Run face detection
    detection = detect_faces(image_rgb)
    steps.append("Face detection completed")

    # Step 2: Run image quality analysis
    quality = analyze_image_quality(image_rgb)
    steps.append("Image quality analysis completed")

    # Step 3: Generate the AI report
    report = generate_analysis_report(detection, quality)
    steps.append("AI report generation attempted")

    return {
        "detection": detection,
        "quality": quality,
        "report": report,
        "steps": steps
    }
