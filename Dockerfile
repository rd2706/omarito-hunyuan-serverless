FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade to PyTorch 2.6.0 (required for HunyuanVideo)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install torch==2.6.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]