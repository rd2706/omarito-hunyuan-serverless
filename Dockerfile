FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8 for RTX 6000 Ada support (sm_89)
RUN pip install --upgrade pip
RUN pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install xformers

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]