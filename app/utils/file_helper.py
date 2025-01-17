import os
import uuid
import asyncio
from pillow_heif import register_heif_opener
from fastapi import UploadFile, HTTPException
from PIL import Image

CHILD_IMAGE_DIRECTORY = "static/child_images/"

async def save_file_async(file: UploadFile) -> str:
    if not os.path.exists(CHILD_IMAGE_DIRECTORY):
        os.makedirs(CHILD_IMAGE_DIRECTORY)

    file_extension = file.filename.split(".")[-1].lower()
    random_name = uuid.uuid4().hex
    random_filename = f"{random_name}.{file_extension}"
    temp_file_path = os.path.join(CHILD_IMAGE_DIRECTORY, f"temp.{file_extension}")
    file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)

    try:
        # Write the file content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        if file_extension == "heic":
            random_filename = f"{random_name}.jpg"
            file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)
            await asyncio.to_thread(convert_heic_to_jpg, temp_file_path, file_path)  # Async conversion
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

async def remove_file_async(file_name: str) -> None:
    try:
        # Run the file removal in a separate thread
        await asyncio.to_thread(os.remove, os.path.join(CHILD_IMAGE_DIRECTORY, file_name))
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
