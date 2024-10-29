# Dockerfile for face_recognition_service
FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn numpy python-multipart pydantic
CMD ["uvicorn", "face_recognition_service:app", "--host", "0.0.0.0", "--port", "8000"]