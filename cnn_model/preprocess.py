import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


def remove_small_folders(dir_path: str, min_files: int) -> None:
    """
    Removes folders in the directory containing fewer than the specified minimum number of files.
    
    Args:
        dir_path (str): Path to the directory to clean.
        min_files (int): Minimum number of files required to retain a folder.
    """
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        return

    for entry in os.scandir(dir_path):
        if entry.is_dir():
            folder_path = entry.path
            file_count = sum(1 for _ in os.scandir(folder_path) if _.is_file())
            if file_count < min_files:
                print(f"Removing folder: {folder_path}")
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    print(f"Error removing {folder_path}: {e}")


def visualize_face_extraction(image_path: str) -> None:
    """
    Visualizes face extraction from a given image path.
    
    Args:
        image_path (str): Path to the image file.
    """
    try:
        face_objs = DeepFace.extract_faces(img_path=image_path, detector_backend="yolov8")
        if face_objs:
            face_img = face_objs[0]["face"]
            face_img = cv2.resize(face_img, (224, 224))

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            original_img = plt.imread(image_path)

            axs[0].imshow(original_img)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(face_img)
            axs[1].set_title("Detected Face")
            axs[1].axis("off")

            plt.show()
            print(f"Detected face shape: {face_img.shape}")
        else:
            print("No face detected in the image.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_image(
    src_image_path: str, dest_image_path: str, min_faces: int, max_faces: int
) -> None:
    """
    Extracts and processes faces from an image.
    
    Args:
        src_image_path (str): Path to the source image.
        dest_image_path (str): Path to save the processed face image.
        min_faces (int): Minimum number of faces required to process the image.
        max_faces (int): Maximum number of faces allowed to process the image.
    """
    try:
        face_objs = DeepFace.extract_faces(img_path=src_image_path, detector_backend="yolov8")
        if min_faces <= len(face_objs) <= max_faces:
            face_image = face_objs[0]["face"]

            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)

            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest_image_path, bgr_face_image)
    except Exception as e:
        print(f"Error processing image {src_image_path}: {e}")


def crop_and_save_faces(
    src_directory: str, dest_directory: str, min_faces: int = 1, max_faces: int = 1
) -> None:
    """
    Extracts and saves cropped faces from all images in a directory.
    
    Args:
        src_directory (str): Source directory containing images.
        dest_directory (str): Destination directory to save cropped faces.
        min_faces (int): Minimum number of faces required in an image for processing.
        max_faces (int): Maximum number of faces allowed in an image for processing.
    """
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_image = {}
        for folder_name in tqdm(os.listdir(src_directory), desc="Processing folders"):
            src_folder_path = os.path.join(src_directory, folder_name)
            dest_folder_path = os.path.join(dest_directory, folder_name)

            if os.path.isdir(src_folder_path):
                os.makedirs(dest_folder_path, exist_ok=True)

                image_files = [
                    f
                    for f in os.listdir(src_folder_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                ]
                for filename in image_files:
                    src_image_path = os.path.join(src_folder_path, filename)
                    dest_image_path = os.path.join(dest_folder_path, filename)
                    future = executor.submit(
                        process_image, src_image_path, dest_image_path, min_faces, max_faces
                    )
                    future_to_image[future] = src_image_path

        for future in tqdm(future_to_image.keys(), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                src_image_path = future_to_image[future]
                print(f"Error processing image {src_image_path}: {e}")
