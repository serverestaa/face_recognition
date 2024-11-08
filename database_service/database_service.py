from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from sqlalchemy.orm import Session
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import cv2
from database import SessionLocal, engine
from models import Face, Base

app = FastAPI()
Base.metadata.create_all(bind=engine)

# Инициализируем модель для кодирования лица
recognition_net = InceptionResnetV1(pretrained='vggface2').eval()


# Зависимость для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Функция для кодирования лица с нормализацией
def encode_face(file_bytes: bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Изменяем размер изображения лица до (160, 160)
    image_resized = cv2.resize(image, (160, 160))

    # Преобразуем изображение в тензор и нормализуем
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        encoding = recognition_net(image_tensor)

    # Нормализуем кодировку лица для единообразия
    encoding = encoding[0].numpy().astype(np.float32)
    encoding = encoding / np.linalg.norm(encoding)

    return encoding.tolist()


# Функция для сохранения лица в базе данных
def save_face_to_db(name: str, encoding: list, db: Session):
    face_encoding = np.array(encoding, dtype=np.float32).tobytes()
    face_record = Face(name=name, face_encoding=face_encoding)
    db.add(face_record)
    db.commit()


@app.post("/add_face/")
async def add_face(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Прочитать изображение
    file_bytes = await file.read()

    # Кодирование лица с нормализацией
    try:
        encoding = encode_face(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при кодировании лица: {str(e)}")

    # Сохранение в базе данных
    try:
        save_face_to_db(name, encoding, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в базе данных: {str(e)}")

    return {"status": "face added", "name": name}


@app.post("/encode_face/")
async def encode_face_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        encoding = encode_face(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding face: {str(e)}")

    return {"encoding": encoding}


@app.get("/get_known_faces/")
async def get_known_faces(db: Session = Depends(get_db)):
    try:
        faces = db.query(Face).all()
        known_faces = [{
            "name": face.name,
            "encoding": np.frombuffer(face.face_encoding, dtype=np.float32).tolist()
        } for face in faces]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving faces: {str(e)}")

    return {"known_faces": known_faces}

