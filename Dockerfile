# Use TensorFlow's official GPU-enabled image
FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /app

# Add NVIDIA GPG key and configure apt to accept repository even with weak signatures
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub || true
RUN apt-get update -o Acquire::AllowInsecureRepositories=true || true

# Install system dependencies for OpenCV and dlib
RUN apt-get install -y --allow-unauthenticated \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    build-essential

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies without exact version constraints
RUN pip install --no-cache-dir face_recognition deepface pillow python-multipart fastapi uvicorn

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]