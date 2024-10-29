from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from facenet_pytorch import MTCNN

app = FastAPI()
mtcnn = MTCNN(keep_all=True)


@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        faces, _ = mtcnn.detect(image)
        return {"faces": [face.tolist() for face in faces] if faces is not None else []}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")