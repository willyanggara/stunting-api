import os
import uuid
from pillow_heif import register_heif_opener
from fastapi import UploadFile, HTTPException
from PIL import Image


CHILD_IMAGE_DIRECTORY = "static/child_images/"  # Direktori untuk menyimpan gambar anak

def save_file(file: UploadFile) -> str:
    """
    Simpan file secara lokal dengan nama file acak.
    Jika file HEIC, konversi menjadi JPG dan ubah ukurannya menjadi 224x224.
    """
    if not os.path.exists(CHILD_IMAGE_DIRECTORY):
        os.makedirs(CHILD_IMAGE_DIRECTORY)

    # Ambil ekstensi file
    file_extension = file.filename.split(".")[-1].lower()

    # Generate nama file acak
    random_name = uuid.uuid4().hex
    random_filename = f"{random_name}.{file_extension}"

    # Tentukan path untuk file sementara
    temp_file_path = os.path.join(CHILD_IMAGE_DIRECTORY, f"temp.{file_extension}")
    file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)

    try:
        # Simpan file sementara
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Cek apakah file adalah HEIC
        if file_extension == "heic":
            random_filename = f"{random_name}.jpg"
            file_path = os.path.join(CHILD_IMAGE_DIRECTORY, random_filename)
            # Jika HEIC, konversi menjadi JPG dan ubah ukurannya menjadi 224x224
            convert_heic_to_jpg(temp_file_path, file_path)

            # Buka gambar JPG yang telah dikonversi dan ubah ukurannya
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Ubah ukuran gambar menjadi 224x224
            image.save(file_path, "JPEG")
        else:
            # Jika bukan HEIC, simpan file langsung dan ubah ukurannya
            image = Image.open(temp_file_path)
            image = image.resize((224, 224))  # Ubah ukuran gambar menjadi 224x224
            image.save(file_path)

        # Hapus file sementara
        os.remove(temp_file_path)

        return random_filename

    except Exception as e:
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
        # Gunakan pillow-heif untuk membuka file HEIC
        image = Image.open(heic_file_path)
        image.verify()  # Memverifikasi gambar (memastikan formatnya valid)

        # Simpan sebagai JPG
        image = Image.open(heic_file_path)  # Reopen to use it after verify()
        image.save(jpg_file_path, "JPEG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error convert file: {str(e)}")