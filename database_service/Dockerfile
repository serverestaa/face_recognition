# Используем базовый образ Python
FROM python:3.8-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем содержимое текущей директории в контейнер
COPY . /app

# Создаем директорию для базы данных
RUN mkdir -p /app/data

# Устанавливаем зависимости
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy numpy \
    python-multipart requests facenet_pytorch torch opencv-python-headless

RUN python -c "from facenet_pytorch import InceptionResnetV1; model = InceptionResnetV1(pretrained='vggface2').eval()"
# Запуск приложения
CMD ["uvicorn", "database_service:app", "--host", "0.0.0.0", "--port", "8000"]