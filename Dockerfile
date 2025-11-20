FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with broader CUDA support for H100/A100 80GB GPUs
RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]