import numpy as np
import cv2
import asyncio
import time
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
DETECTION_URL = "http://localhost:8001/detect_faces/"
ENCODING_URL = "http://localhost:8003/encode_face/"
KNOWN_FACES_URL = "http://localhost:8003/get_known_faces/"
RECOGNITION_URL = "http://localhost:8004/recognize_face/"


@app.get("/video_feed/")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


async def gen_frames():
    video_capture = cv2.VideoCapture(0)
    retry_count = 0
    max_retries = 10

    while not video_capture.isOpened() and retry_count < max_retries:
        print("Attempting to open webcam... (Retry)", retry_count + 1)
        time.sleep(1)
        video_capture = cv2.VideoCapture(0)
        retry_count += 1

    if not video_capture.isOpened():
        print("Cannot open webcam after retries")
        return

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    print("Warning: No frame captured from the webcam.")
                    await asyncio.sleep(0.1)
                    continue

                _, img_encoded = cv2.imencode('.jpg', frame)
                files = {'file': img_encoded.tobytes()}

                try:
                    print("Attempting face detection")
                    detect_response = await client.post(DETECTION_URL, files=files)
                    detect_response.raise_for_status()
                    faces = detect_response.json().get("faces", [])
                    print(f"Faces detected: {len(faces)}")
                except Exception as e:
                    print(f"Face detection error: {e}")
                    continue

                if faces:
                    for face_coords in faces:
                        x1, y1, x2, y2 = map(int, face_coords)
                        face_image = frame[y1:y2, x1:x2]
                        face_image = cv2.resize(face_image, (160, 160))
                        _, face_encoded = cv2.imencode('.jpg', face_image)
                        try:
                            face_response = await client.post(ENCODING_URL, files={'file': face_encoded.tobytes()})
                            face_response.raise_for_status()
                            face_encoding = face_response.json().get("encoding")
                            print("Face encoding retrieved.")
                        except Exception as e:
                            print(f"Face encoding error: {e}")
                            continue

                        try:
                            print("Fetching known faces from database")
                            known_faces_response = await client.get(KNOWN_FACES_URL)
                            known_faces_response.raise_for_status()
                            known_faces = known_faces_response.json().get("known_faces", [])
                        except Exception as e:
                            print(f"Known faces retrieval error: {e}")
                            continue

                        try:
                            print("Attempting face recognition")
                            known_encodings = [
                                np.array(face["encoding"], dtype=np.float32) / np.linalg.norm(face["encoding"]) for face
                                in known_faces]
                            known_names = [face["name"] for face in known_faces]

                            # Нормализуем encoding нового лица
                            face_encoding = np.array(face_encoding, dtype=np.float32) / np.linalg.norm(face_encoding)
                            face_encoding = face_encoding.tolist()
                            known_encodings = [enc.tolist() for enc in known_encodings]

                            recognition_resp = await client.post(
                                RECOGNITION_URL,
                                json={
                                    "encoding": face_encoding,
                                    "known_encodings": known_encodings,
                                    "known_names": known_names
                                }
                            )
                            recognition_resp.raise_for_status()
                            matches = recognition_resp.json().get("matches", [])
                            name = matches[0] if matches else "Unknown"
                            print(f"Recognition result: {name}")

                            # Отрисовка результата на кадре
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        except Exception as e:
                            print(f"Face recognition error: {e}")
                            continue

                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Frame encoding error: {e}")

        finally:
            video_capture.release()