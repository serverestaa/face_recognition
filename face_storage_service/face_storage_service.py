import grpc
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.face_recognition_service import face_pb2, face_pb2_grpc

app = FastAPI()

DATABASE_URL = "sqlite:///./faces.db"
Base = declarative_base()


class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    face_encoding = Column(LargeBinary)


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


@app.post("/upload_face/")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()

    # Connect to Face Recognition Service
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = face_pb2_grpc.FaceRecognitionStub(channel)
        request = face_pb2.EncodeFaceRequest(image=file_bytes)

        try:
            response = stub.EncodeFace(request)
            encoding = np.array(response.encoding, dtype=np.float32)
            add_face_to_db(name, encoding)
            return {"status": "success", "name": name}
        except grpc.RpcError as e:
            raise HTTPException(status_code=400, detail=e.details())


def add_face_to_db(name: str, face_encoding: np.ndarray):
    session = SessionLocal()
    try:
        face = Face(name=name, face_encoding=face_encoding.tobytes())
        session.add(face)
        session.commit()
    finally:
        session.close()


@app.get("/get_known_faces/")
def get_known_faces():
    session = SessionLocal()
    try:
        faces = session.query(Face).all()
        known_faces = [{"name": face.name, "encoding": np.frombuffer(face.face_encoding, dtype=np.float32).tolist()} for face in faces]
        return known_faces
    finally:
        session.close()