# Face Verification

This project implements a face verification system using a combination of machine learning, image processing, and gRPC services. The system provides functionalities to train models, detect faces, encode them, and stream video with real-time face verification.

**Paper Report** : [Baisbay_Shildebayev.pdf](https://github.com/user-attachments/files/17879113/Baisbay_Shildebayev.pdf)


https://github.com/user-attachments/assets/e5a0dcc7-41d5-4bd2-b81a-7bf3f493cdc9

---

## Features

1. **Face Encoding and Recognition**:
   - Encodes face images using a pre-trained neural network.
   - Matches encoded faces against known identities stored in a database.

2. **Real-Time Face Detection**:
   - Processes video streams and identifies faces in real time.
   - Integrates with gRPC and FastAPI to expose APIs for face recognition and video streaming.

3. **Model Training**:
   - Train deep learning models using triplet loss for face encoding.
   - Provides scripts for preprocessing datasets and training models.

4. **Database Integration**:
   - Stores face encodings and corresponding metadata in a SQLite database.
   - Exposes API endpoints to retrieve and manage known faces.

---

## Project Structure

| File                       | Description                                                             |
|----------------------------|-------------------------------------------------------------------------|
| `inference.py`             | Handles inference for face encoding and detection.                     |
| `main.py`                  | Entry point for various services and utilities in the system.           |
| `model_training.py`        | Script for training the deep learning model using triplet loss.         |
| `preprocess.py`            | Prepares and processes datasets for model training.                    |
| `utils.py`                 | Contains utility functions like dataset loaders and triplet loss computation. |
| `face_storage_service.py`  | FastAPI-based service for storing and managing face encodings.          |
| `streaming_service.py`     | Handles video streaming and real-time face recognition.                |
| `service.py`               | gRPC server for encoding faces and providing recognition capabilities. |
| `face_pb2.py`              | gRPC protocol buffer definitions for face recognition.                 |
| `face_pb2_grpc.py`         | gRPC stubs and server interfaces generated from protocol buffer definitions. |

---

## Dataset used for training and validation:

The LFW dataset contains 13,233 images of faces collected from the web. This dataset consists of the 5749 identities with 1680 people with two or more images. 
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Dataset used during inference:

12 classes with 6-9 images which gives in total of 94 images. 
https://drive.google.com/drive/folders/17cku0kxf-tmfZyLz3HGCKFQ4mbn61Xwr?usp=sharing

---

## Prerequisites

- **Python** 3.8+
- `pipenv` or `virtualenv` for dependency management
- A CUDA-enabled GPU (for training and inference)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/facerecognition
   cd facerecognition
   ```

2. **Set up the Python environment and install dependencies**:
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Install additional requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the SQLite database**:
   ```bash
   sqlite3 faces.db < schema.sql
   ```

---

## Usage

1. **Start the gRPC Server**:
   ```bash
   python service.py
   ```

2. **Start the API Services**:
   ```bash
   uvicorn face_storage_service:app --host 0.0.0.0 --port 8000
   ```

3. **Start the Video Streaming**:
   ```bash
   uvicorn streaming_service:app --host 0.0.0.0 --port 8080
   ```

4. **Train the Model**:
   ```bash
   python main.py
   ```

---

## API Endpoints

### Face Storage Service:
- **POST `/upload_face/`**: Upload a face image and store its encoding.
- **GET `/get_known_faces/`**: Retrieve all known faces and their encodings.

### Streaming Service:
- **GET `/video_feed`**: Access the real-time video feed with face recognition overlays.

---

## Acknowledgments

This project utilizes:
- **PyTorch** for model development and inference.
- **FastAPI** for API creation.
- **gRPC** for efficient client-server communication.

---

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
