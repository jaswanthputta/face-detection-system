
import cv2
import numpy as np


def detect_faces(image_rgb):
    """Detect faces and return the annotated image and bounding boxes."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load the face detection model.")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result = image_rgb.copy()
    boxes = []

    for x, y, w, h in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 220, 100),
            3
        )
        boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })

    return {
        "face_count": len(boxes),
        "bounding_boxes": boxes,
        "annotated_image": result
    }


def analyze_image_quality(image_rgb):
    """Estimate image sharpness, brightness, and resolution."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(
        gray, cv2.CV_64F
    ).var())

    brightness = float(np.mean(gray))
    height, width = gray.shape

    if sharpness < 100:
        sharpness_label = "Potentially blurry"
    else:
        sharpness_label = "Reasonably sharp"

    if brightness < 70:
        brightness_label = "Dark"
    elif brightness > 190:
        brightness_label = "Bright"
    else:
        brightness_label = "Moderate"

    return {
        "width": int(width),
        "height": int(height),
        "sharpness_score": round(sharpness, 2),
        "sharpness_assessment": sharpness_label,
        "average_brightness": round(brightness, 2),
        "brightness_assessment": brightness_label
    }
