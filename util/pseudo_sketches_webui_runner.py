import os
import cv2
import requests
import json

import base64
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
from io import BytesIO


class WebuiAPI:
    def __init__(self, url="http://127.0.0.1:7860") -> None:
        self.url = url
        self.__set_model()

    def get_available_loras(self) -> requests.Response:
        url = "/".join([self.url, "sdapi", "v1", "loras"])
        
        response = requests.get(url=url)
        # Check the response (optional)
        if response.status_code == 200:
            pass
        else:
            print(f"Request failed with status code {response.status_code}.")
            
        return response
    
    def generate_image(
            self, 
            prompt="cafe", steps=7, cfg_scale=2, 
            width=512, height=512, 
            sampler="DPM++ SDE Karras",
            batch_size=4,
            sketch=None,
            seed=22022
    ):
        url = "/".join([self.url, "sdapi", "v1", "txt2img"])

        # Prepare the headers
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        if sketch is None:
            prompt = f"ais-lineart, a {prompt}, <lora:Line_Art_SDXL:1.2>"

        # Prepare the data payload
        data = {
            "prompt": prompt,
            "negative_prompt":
                "bad, worst",
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "sampler_index": sampler,
            "batch_size": batch_size,
        }

        # 3CPM
        if sketch is not None:
            # Read Image in RGB order
            sketch = self.pil_to_cv2(sketch)

            # Encode into PNG and send to ControlNet
            retval, bytes = cv2.imencode('.png', sketch)
            encoded_image = base64.b64encode(bytes).decode('utf-8')
            
            data["alwayson_scripts"] = {
                "controlnet": {
                    "args": [{
                        "input_image": encoded_image,
                        "model": "diffusers_xl_canny_full [2b69fca4]",
                        "weight": 0.7,
                        "resize_mode": 0
                    }]
                }
            }
            data["prompt"] = prompt
            data["width"] = sketch.shape[0]
            data["height"] = sketch.shape[1]
            
        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check the response (optional)
        if response.status_code == 200:
            return self.response_to_pil(response)
        else:
            print(f"Request failed with status code {response.status_code}.")
            return None

    @staticmethod
    def response_to_pil(response):
        # Assume 'response' is your requests response
        response_json = response.json()

        images = []

        for i in range(len(response_json['images'])):
            # Extract base64 encoded string (assuming the first image in the list)
            encoded_image = response_json['images'][i]

            # Decode the base64 string
            image_data = base64.b64decode(encoded_image)

            # Convert to an image
            image = Image.open(BytesIO(image_data))
            cv_image = WebuiAPI.pil_to_cv2(image)

            images.append(cv_image)
        
        return images

    def __set_model(self, model_name="dreamshaperXL_v21TurboDPMSDE.safetensors", clip_skip=1):
        url = "/".join([self.url, "sdapi", "v1", "options"])

        option_payload = {
            "sd_model_checkpoint": model_name,
            "CLIP_stop_at_last_layers": clip_skip,
        }

        response = requests.post(url=url, json=option_payload)
        
        # Check the response (optional)
        if response.status_code == 200:
            pass
        else:
            print(f"Request failed with status code {response.status_code}.")
            
    @staticmethod
    def pil_to_cv2(pil_image):
        # Convert PIL Image to numpy array
        numpy_image = np.array(pil_image)

        # Convert from RGB to BGR (OpenCV format)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        return opencv_image