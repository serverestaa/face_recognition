import cv2
import numpy as np
import grpc
from concurrent import futures
import torch
from torchvision import transforms
from PIL import Image

from face_recognition_service import face_pb2, face_pb2_grpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model_scripted_EfficientNetv3_10epochs.pt", map_location=device)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_face_encoding(face_image):
    face_image_tensor = transform(Image.fromarray(face_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoding = model(face_image_tensor).cpu().numpy().flatten()
    encoding = encoding / np.linalg.norm(encoding)
    return encoding


class FaceRecognitionServicer(face_pb2_grpc.FaceRecognitionServicer):
    def EncodeFace(self, request, context):
        np_img = np.frombuffer(request.image, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(detected_faces) == 0:
            context.set_details("No faces found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return face_pb2.EncodeFaceResponse()

        x, y, w, h = detected_faces[0]
        face_image = image[y:y + h, x:x + w]
        encoding = get_face_encoding(face_image)
        return face_pb2.EncodeFaceResponse(encoding=encoding.tolist())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    face_pb2_grpc.add_FaceRecognitionServicer_to_server(FaceRecognitionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Face Recognition Service running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()