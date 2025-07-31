# Use the specified NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.06-py3



RUN apt-get update && apt-get install -y \
 ffmpeg \
 libgl1 && \
 rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


EXPOSE 5000


CMD ["python", "app.py"]
