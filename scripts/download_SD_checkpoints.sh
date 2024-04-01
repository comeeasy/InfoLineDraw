#! /bin/bash

## Take path of "stable-sd-webui directory" as an argument 
# Check if an argument is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_stable-sd-webui_directory>"
    exit 1
fi

# Assign the first argument to a variable
SD_WEBUI_PATH="$1"

# Check if the stable-sd-webui directory exists
if [ ! -d "$SD_WEBUI_PATH" ]; then
    echo "The path $SD_WEBUI_PATH does not exist."
    exit 1
fi

# Define the target directories
STABLEDIFFUSION_MODEL_PATH="${SD_WEBUI_PATH}/models/Stable-diffusion"
CONTROLNET_MODEL_PATH="${SD_WEBUI_PATH}/models/ControlNet"
LORA_MODEL_PATH="${SD_WEBUI_PATH}/models/Lora"

# Download https://drive.google.com/file/d/1PhWkky-4mMJqqgCtr5IlcQZdvIdex2T8/view?usp=sharing
# to "stable-sd-webui/models/StableDiffusion" using gdwon. And Path existence must be checked
# Check and create StableDiffusion model directory if it doesn't exist
if [ -d "$STABLEDIFFUSION_MODEL_PATH" ]; then
    echo "Downloading Stable Diffusion model..."
    gdown "https://drive.google.com/uc?id=1PhWkky-4mMJqqgCtr5IlcQZdvIdex2T8" -O "${STABLEDIFFUSION_MODEL_PATH}/dreamshaperXL_v21TurboDPMSDE.safetensors"
else
    echo "Directory $STABLEDIFFUSION_MODEL_PATH does not exist. Skipping download."
fi


# Download https://drive.google.com/file/d/1-0npDwTU4YSe5JlOly6Va1fuz_tjlZzh/view?usp=sharing
# to "stable-sd-webui/models/ControlNet" using gdwon. And Path existence must be checked
# Check and create ControlNet model directory if it doesn't exist
if [ -d "$CONTROLNET_MODEL_PATH" ]; then
    echo "Downloading ControlNet model..."
    gdown "https://drive.google.com/uc?id=1-0npDwTU4YSe5JlOly6Va1fuz_tjlZzh" -O "${CONTROLNET_MODEL_PATH}/diffusers_xl_canny_full.safetensors"
else
    echo "Directory $CONTROLNET_MODEL_PATH does not exist. Skipping download."
fi

# Download https://drive.google.com/file/d/1Gy4PVhuDVfztdTFimMtc7hfngmL2stm2/view?usp=sharing
# to "stable-sd-webui/models/Lora" using gdwon. And Path existence must be checked
if [ -d "$LORA_MODEL_PATH" ]; then
    echo "Downloading Lora model..."
    gdown "https://drive.google.com/uc?id=1-0npDwTU4YSe5JlOly6Va1fuz_tjlZzh" -O "${LORA_MODEL_PATH}/Line_Art_SDXL.safetensors"
else
    echo "Directory $LORA_MODEL_PATH does not exist. Skipping download."
fi



