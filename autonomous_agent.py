
import os
import json
import requests
from dotenv import load_dotenv

from analysis_tools import detect_faces, analyze_image_quality

load_dotenv(
    os.path.join(os.path.dirname(__file__), ".env")
)

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_faces",
            "description": (
                "Detect faces in the uploaded image and return "
                "the number of detected faces and bounding boxes."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image_quality",
            "description": (
                "Measure the uploaded image's sharpness and "
                "brightness and report its dimensions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]


def call_agent(messages, available_tools):
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing from your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": available_tools,
        "tool_choice": "auto",
        "max_tokens": 1200,
        "temperature": 0.2,
        "stream": True
    }

    content_parts = []
    tool_calls = {}

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(10, 120)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            chunk = json.loads(data)
            choices = chunk.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})

            content = delta.get("content")
            if content:
                content_parts.append(content)

            for item in delta.get("tool_calls", []):
                index = item["index"]

                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }

                current = tool_calls[index]

                if item.get("id"):
                    current["id"] = item["id"]

                function = item.get("function", {})

                if function.get("name"):
                    current["function"]["name"] += (
                        function["name"]
                    )

                if function.get("arguments"):
                    current["function"]["arguments"] += (
                        function["arguments"]
                    )

    return {
        "content": "".join(content_parts),
        "tool_calls": [
            tool_calls[index]
            for index in sorted(tool_calls)
        ]
    }


def run_autonomous_agent(image_rgb):
    messages = [
        {
            "role": "system",
            "content": (
                "You are FaceAI Pro, an image-analysis agent. "
                "You can use face detection and image quality "
                "tools. Use both tools for a complete image "
                "analysis. Choose which tool to call first, "
                "inspect each result, and decide whether another "
                "tool call is needed. Never identify people or "
                "infer their identity. Do not invent measurements. "
                "When the tools are complete, provide a clear "
                "report with an analysis summary, face detection "
                "results, image quality assessment, and practical "
                "image-quality suggestions. Explain that brightness "
                "and sharpness are estimates."
            )
        },
        {
            "role": "user",
            "content": (
                "Analyze the uploaded image using the available "
                "tools and prepare the final report."
            )
        }
    ]

    results = {}
    steps = []
    executed_tools = set()
    max_rounds = 4

    for round_number in range(max_rounds):
        available_tools = [
            tool for tool in TOOLS
            if tool["function"]["name"] not in executed_tools
        ]

        response = call_agent(messages, available_tools)
        calls = response["tool_calls"]

        if not calls:
            report = response["content"].strip()

            if not report:
                report = (
                    "The AI model returned no final report. "
                    "Please try again."
                )

            return {
                "detection": results.get("detect_faces"),
                "quality": results.get("analyze_image_quality"),
                "report": report,
                "steps": steps,
                "tool_calls": list(executed_tools)
            }

        assistant_message = {
            "role": "assistant",
            "content": response["content"] or None,
            "tool_calls": calls
        }
        messages.append(assistant_message)

        for call in calls:
            name = call["function"]["name"]
            call_id = call["id"]

            if name not in {
                "detect_faces",
                "analyze_image_quality"
            }:
                tool_result = {
                    "error": "Tool is not permitted."
                }

            elif name in executed_tools:
                tool_result = {
                    "error": "This tool has already been executed."
                }

            else:
                try:
                    if name == "detect_faces":
                        result = detect_faces(image_rgb)

                        results[name] = result

                        tool_result = {
                            "face_count": result["face_count"],
                            "bounding_boxes": result[
                                "bounding_boxes"
                            ]
                        }

                        steps.append("Face detection completed")

                    else:
                        result = analyze_image_quality(image_rgb)

                        results[name] = result
                        tool_result = result

                        steps.append(
                            "Image quality analysis completed"
                        )

                    executed_tools.add(name)

                except Exception as error:
                    tool_result = {
                        "error": (
                            f"{name} failed: {str(error)}"
                        )
                    }

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_result)
            })

    return {
        "detection": results.get("detect_faces"),
        "quality": results.get("analyze_image_quality"),
        "report": (
            "The agent reached its tool-call limit before "
            "producing a final report. Please try again."
        ),
        "steps": steps,
        "tool_calls": list(executed_tools)
    }
