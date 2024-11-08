from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()


# Define the request model
class RecognitionRequest(BaseModel):
    encoding: List[float]
    known_encodings: List[List[float]]
    known_names: List[str]
    tolerance: float = 0.6


@app.post("/recognize_face/")
def recognize_face(request: RecognitionRequest):
    try:
        encoding_to_check = np.array(request.encoding, dtype=np.float32)
        encoding_to_check = encoding_to_check / np.linalg.norm(encoding_to_check)
        normalized_encodings = [np.array(enc, dtype=np.float32) / np.linalg.norm(enc) for enc in
                                request.known_encodings]
        distances = np.linalg.norm(np.array(normalized_encodings) - encoding_to_check, axis=1)

        matches = list(distances <= request.tolerance)
        matched_names = [request.known_names[i] for i, match in enumerate(matches) if match]

        return {"matches": matches,
                "matched_names": matched_names}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid encoding or known_encodings format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recognition: {str(e)}")
