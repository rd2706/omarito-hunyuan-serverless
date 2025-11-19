FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install in stages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir diffusers transformers huggingface_hub runpod
RUN pip install --no-cache-dir opencv-python pillow imageio accelerate safetensors

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]