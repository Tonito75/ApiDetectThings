FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    gcc g++ git wget \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLOv5 (runtime de MegaDetector v5)
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5 \
    && pip install --no-cache-dir -r /app/yolov5/requirements.txt

# Poids MegaDetector v5a
RUN wget -q --show-progress -O /app/md_v5a.0.0.pt \
    https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt

# Code
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
