# Use an NVIDIA CUDA base image that provides the needed GPU libraries
FROM nvidia/cuda:11.0.3-base

# Retrieve the missing NVIDIA public key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install Python 3, pip, and other dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip cmake build-essential libgl1 libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]