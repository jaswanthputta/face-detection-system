
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(
    os.path.join(os.path.dirname(__file__), ".env")
)


def generate_analysis_report(detection, quality):
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        return (
            "NVIDIA API key is missing. "
            "Check your .env file."
        )

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        analysis_data = {
            "faces_detected": detection["face_count"],
            "face_locations": detection["bounding_boxes"],
            "image_width": quality["width"],
            "image_height": quality["height"],
            "sharpness_score": quality["sharpness_score"],
            "sharpness_assessment": quality["sharpness_assessment"],
            "average_brightness": quality["average_brightness"],
            "brightness_assessment": quality["brightness_assessment"]
        }

        prompt = f"""
Analyze these computer vision results:

{json.dumps(analysis_data, indent=2)}

Create a report with these sections:
1. Analysis Summary
2. Face Detection Results
3. Image Quality Assessment
4. Suggestions for Improving Image Quality

Use simple English.
Only use the supplied measurements.
Do not identify people or invent facts about them.
Explain that brightness and sharpness are estimates.
Clearly state if no faces were detected.
"""

        response = client.chat.completions.create(
              model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful computer vision "
                        "report assistant. Explain results "
                        "clearly and accurately."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        report = response.choices[0].message.content

        return report or "The model returned an empty report."

    except Exception as error:
        return (
            "The NVIDIA AI report could not be generated. "
            f"Error: {error}"
        )
