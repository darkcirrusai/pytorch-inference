import shutil
from fastapi import UploadFile
from pathlib import Path
import base64


# function to save uploaded files
def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


# function to convert base64 img to jpeg
def convert_b64_jpeg(encoded_img, save_path):
    image_data = base64.b64decode(encoded_img)
    with open(save_path, 'wb') as f:
        f.write(image_data)
