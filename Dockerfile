FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install requirements  
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Note: Base model will be downloaded on first inference to avoid build timeouts

# Copy handler
COPY handler.py .

# Set the default command
CMD ["python", "handler.py"]