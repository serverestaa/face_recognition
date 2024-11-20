import cv2
import grpc
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from face_recognition_service import face_pb2, face_pb2_grpc

app = FastAPI()


def gen_frames():
    video_capture = cv2.VideoCapture(0)
    known_faces = load_known_faces()

    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in detected_faces:
                face_image = frame[y:y + h, x:x + w]
                _, buffer = cv2.imencode(".jpg", face_image)
                face_bytes = buffer.tobytes()

                with grpc.insecure_channel("localhost:50051") as channel:
                    stub = face_pb2_grpc.FaceRecognitionStub(channel)
                    request = face_pb2.EncodeFaceRequest(image=face_bytes)

                    try:
                        response = stub.EncodeFace(request)
                        encoding = np.array(response.encoding, dtype=np.float32)
                        name = compare_with_known_faces(known_faces, encoding)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    except grpc.RpcError:
                        pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        video_capture.release()


def load_known_faces():
    import requests
    response = requests.get("http://localhost:8000/get_known_faces/")
    if response.status_code == 200:
        return response.json()
    return []


def compare_with_known_faces(known_faces, face_encoding, tolerance=0.35):
    for face in known_faces:
        encoding = np.array(face["encoding"], dtype=np.float32)
        distance = np.linalg.norm(encoding - face_encoding)
        if distance <= tolerance:
            return face["name"]
    return "Unknown"


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")