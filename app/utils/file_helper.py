import os
import uuid
from pillow_heif import register_heif_opener
from fastapi import UploadFile, HTTPException
from PIL import Image

CHILD_IMAGE_DIRECTORY = "static/child_images/"

def save_file(file: UploadFile) -> str:
    if not os.path.exists(CHILD_IMAGE_DIRECTORY):
        os.makedirs(CHILD_IMAGE_DIRECTORY)

    file_extension = file.filename.split(".")[-1].lower()
    random_name = uuid.uuid4().hex
    random_filename = f"{random_name}.{file_extension}"
    temp_file_path = os.path.join(CHILD_IMAGE_DIRECTORY, f"temp.{file_extension}")
    file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        if file_extension == "heic":
            random_filename = f"{random_name}.jpg"
            file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)
            convert_heic_to_jpg(temp_file_path, file_path)
            image = Image.open(file_path)
        else:
            image = Image.open(temp_file_path)

        image = image.resize((224, 224))
        image.save(file_path)

        os.remove(temp_file_path)
        return random_filename

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def remove_file(file_name: str) -> None:
    try:
        os.remove(os.path.join(CHILD_IMAGE_DIRECTORY, file_name))
    except OSError:
        pass

def convert_heic_to_jpg(heic_file_path, jpg_file_path):
    try:
        register_heif_opener()
        image = Image.open(heic_file_path)
        image.verify()
        image = Image.open(heic_file_path)
        image.save(jpg_file_path, "JPEG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting file: {str(e)}")

