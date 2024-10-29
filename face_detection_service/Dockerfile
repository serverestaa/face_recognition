FROM python:3.8-slim

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn opencv-python-headless numpy facenet-pytorch torch python-multipart

CMD ["uvicorn", "face_detection_service:app", "--host", "0.0.0.0", "--port", "8000"]