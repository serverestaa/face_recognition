import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./faces.db"
Base = declarative_base()


# Model for storing face encodings
class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    face_encoding = Column(LargeBinary)


# Database connection setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Load MTCNN for face detection and InceptionResnetV1 for face encoding extraction
mtcnn = MTCNN(keep_all=True)
recognition_net = InceptionResnetV1(pretrained='vggface2').eval()


# Function to extract face encodings using InceptionResnetV1
def get_face_encoding(face_image):
    face_image_resized = cv2.resize(face_image, (160, 160))
    face_image_resized = torch.tensor(face_image_resized).permute(2, 0, 1).float() / 255.0
    face_image_resized = face_image_resized.unsqueeze(0)

    with torch.no_grad():
        vec = recognition_net(face_image_resized)
    return vec[0].numpy().astype(np.float32)  # Ensure float32 for consistency


# Function to detect faces using MTCNN
def extract_face_mtcnn(image):
    faces, _ = mtcnn.detect(image)
    face_images = []
    face_locations = []

    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = [int(coord) for coord in face]
            face_image = image[y1:y2, x1:x2]
            face_images.append(face_image)
            face_locations.append((x1, y1, x2, y2))
    return face_images, face_locations


# Upload new face via API
@app.post("/upload_face/")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)) -> dict:
    file_bytes = await file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect faces in the image using MTCNN
    faces, _ = extract_face_mtcnn(image)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face found in the image")

    # Extract 128D face encoding using InceptionResnetV1
    face_encoding = get_face_encoding(faces[0])

    # Ensure face encoding has the correct length (512 for InceptionResnetV1)
    if len(face_encoding) != 512:
        raise HTTPException(status_code=400, detail="Error: incorrect face encoding length")

    # Add face to database
    add_face_to_db(name, face_encoding)

    return {"status": "success", "name": name}


# Function to add face to the database with length check
def add_face_to_db(name: str, face_encoding: np.ndarray):
    if len(face_encoding) != 512:
        raise ValueError(f"Incorrect face encoding length, expected 512 but got {len(face_encoding)}")

    session = SessionLocal()
    try:
        face = Face(name=name, face_encoding=face_encoding.tobytes())
        session.add(face)
        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# Function to clean invalid faces from the database
def clean_invalid_faces_from_db():
    session = SessionLocal()
    try:
        faces = session.query(Face).all()

        for face in faces:
            encoding = np.frombuffer(face.face_encoding, dtype=np.float32)
            if len(encoding) != 512:  # Delete invalid encoding entries
                print(f"Deleting invalid face with encoding length: {len(encoding)}")
                session.delete(face)

        session.commit()
    finally:
        session.close()


# Function to retrieve all faces from the database
def get_known_faces_from_db():
    session = SessionLocal()
    known_face_encodings = []
    known_face_names = []

    try:
        faces = session.query(Face).all()

        for face in faces:
            encoding = np.frombuffer(face.face_encoding, dtype=np.float32)
            if len(encoding) == 512:  # Ensure valid encoding length
                known_face_encodings.append(encoding)
                known_face_names.append(face.name)
            else:
                print(f"Warning: Found face with incorrect encoding length: {len(encoding)}")

    finally:
        session.close()

    return known_face_encodings, known_face_names

# Function to compare face encodings
def compare_face_encodings(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    known_face_encodings = [encoding / np.linalg.norm(encoding) for encoding in known_face_encodings]
    face_encoding_to_check = face_encoding_to_check / np.linalg.norm(face_encoding_to_check)

    distances = np.linalg.norm(np.array(known_face_encodings) - face_encoding_to_check, axis=1)
    return list(distances <= tolerance), distances

# Function to generate video stream with face recognition
def gen_frames():
    clean_invalid_faces_from_db()
    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
            known_face_encodings, known_face_names = get_known_faces_from_db()

            faces, face_locations = extract_face_mtcnn(small_frame)
            face_encodings = []
            for face in faces:
                face_encoding = get_face_encoding(face)
                if len(face_encoding) == 512:
                    face_encodings.append(face_encoding)

            face_names = []
            for face_encoding in face_encodings:
                if known_face_encodings:
                    matches, distances = compare_face_encodings(known_face_encodings, face_encoding)
                    name = "Unknown"
                    if True in matches:
                        match_index = matches.index(True)
                        name = known_face_names[match_index]
                        print(f"Face recognized: {name}, distance: {distances[match_index]}")

                    face_names.append(name)
                else:
                    face_names.append("Unknown")

            for (startX, startY, endX, endY), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, endY - 35), (endX, endY), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (startX + 6, endY - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        video_capture.release()

# Route for viewing video stream
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
