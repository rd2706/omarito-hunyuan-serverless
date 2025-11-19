FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the base model to reduce cold start time
RUN python -c "from diffusers import HunyuanVideoPipeline; HunyuanVideoPipeline.from_pretrained('hunyuanvideo-community/HunyuanVideo')"

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]