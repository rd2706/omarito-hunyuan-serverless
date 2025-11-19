FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TESTED PyTorch 2.5.1 + xformers 0.0.29 + CUDA 12.4
RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]