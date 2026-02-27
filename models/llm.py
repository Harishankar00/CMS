import requests
import base64
import json
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise Exception("API key not found in .env file")

def analyze_classroom(photo_path):
    """
    Analyze a classroom image using NVIDIA API and return the response.

    Args:
        photo_path (str): Path to the classroom image file.

    Returns:
        dict: Parsed JSON response from the API.

    Raises:
        Exception: If the API call fails or the response is invalid.
    """
    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"

    # Read and encode the image in Base64
    try:
        with open(photo_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        raise Exception(f"Image file not found: {photo_path}")
    except Exception as e:
        raise Exception(f"Error encoding image file: {e}")
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # Structured classroom analysis prompt
    analysis_prompt = """Analyze this classroom/training session image and provide a detailed engagement report.
Use <br> tags for line breaks. Structure your response as follows:

<b>Scene Overview</b><br>
Describe the setting, infrastructure, number of people visible, and seating arrangement.<br><br>

<b>Engagement Assessment</b><br>
Evaluate the attention and engagement level of the people present:
- Body posture and orientation (facing forward, leaning, slouching)
- Eye contact and focus direction
- Devices or materials in use (laptops, phones, notebooks)
- Signs of active participation or disengagement<br><br>

<b>Classroom Environment</b><br>
Comment on lighting, space utilization, and any factors that could affect learning.<br><br>

<b>Recommendations</b><br>
Provide 2-3 actionable suggestions to improve engagement and the learning environment."""

    # Payload for the API request
    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.30,
        "top_p": 0.70,
        "stream": False
    }
    
    # Make the API request
    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    
    # Parse and return the response
    try:
        return response.json()
    except json.JSONDecodeError:
        raise Exception("Failed to decode API response as JSON")
