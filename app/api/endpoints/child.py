from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from typing import List

from app.models import child as child_model, child_image as child_image_model
from app.schemas import child as child_schema, child_image as child_image_schema
from app.api import deps
from app.utils.file_helper import save_file, remove_file
from pydantic import parse_obj_as
from app.core.config import settings

router = APIRouter()

@router.get("/summary", response_model=child_schema.SummaryChildren)
def get_summary(db: Session = Depends(deps.get_db)):
    # Ensure data exists
    total_data = db.query(child_model.Child).count()
    if total_data == 0:
        return {
            "total_data": 0,
            "total_stunting": 0,
            "total_not_stunting": 0,
            "total_male": 0,
            "total_female": 0,
            "average_height": 0,
            "average_weight": 0,
            "average_age": 0,
        }

    # Aggregated counts and averages
    total_stunting = db.query(func.count()).filter(child_model.Child.is_stunting == True).scalar()
    total_not_stunting = db.query(func.count()).filter(child_model.Child.is_stunting == False).scalar()
    total_male = db.query(func.count()).filter(child_model.Child.gender == "male").scalar()
    total_female = db.query(func.count()).filter(child_model.Child.gender == "female").scalar()

    average_height = db.query(func.avg(child_model.Child.height)).scalar()
    average_weight = db.query(func.avg(child_model.Child.weight)).scalar()
    average_age = db.query(func.avg(child_model.Child.age)).scalar()

    # Ensure averages are floats, handle None, and round to 2 decimal places
    average_height = round(float(average_height), 2) if average_height is not None else 0.0
    average_weight = round(float(average_weight), 2) if average_weight is not None else 0.0
    average_age = round(float(average_age), 2) if average_age is not None else 0.0

    return {
        "total_data": total_data,
        "total_stunting": total_stunting,
        "total_not_stunting": total_not_stunting,
        "total_male": total_male,
        "total_female": total_female,
        "average_height": average_height,
        "average_weight": average_weight,
        "average_age": average_age,
    }


@router.post("", response_model=child_schema.Child)
async def create_child(
        name: str = Form(...),
        gender: str = Form(...),
        height: float = Form(...),
        weight: float = Form(...),
        age: int = Form(...),
        is_stunting: bool = Form(...),
        images: List[UploadFile] = File(None),
        db: Session = Depends(deps.get_db)
):
    child_data = {
        "name": name,
        "gender": gender,
        "height": height,
        "weight": weight,
        "age": age,
        "is_stunting": is_stunting
    }
    child = parse_obj_as(child_schema.ChildCreate, child_data)

    db_child = child_model.Child(**child.dict())
    db.add(db_child)
    db.commit()
    db.refresh(db_child)

    if images:
        for image in images:
            filepath = save_file(image)

            db_image = child_image_model.ChildImage(
                child_id=db_child.id,
                original_filename=image.filename,
                image_url=filepath
            )
            db.add(db_image)

    db.commit()
    db.refresh(db_child)
    return db_child

@router.get("/{child_id}", response_model=child_schema.Child)
def get_child(child_id: int, request: Request, db: Session = Depends(deps.get_db)):
    db_child = (
        db.query(child_model.Child)
        .options(joinedload(child_model.Child.images))
        .filter(child_model.Child.id == child_id)
        .first()
    )
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Update image_url to full URL
    base_url = str(request.base_url)
    for image in db_child.images:
        image.image_url = f"{base_url}static/child_images/{image.image_url}"

    return db_child


from fastapi import Query

@router.get("", response_model=child_schema.PaginatedChildren)
def list_children(
    page: int = 1,
    page_size: int = 10,
    search: str = Query(None, description="Search for child name"),
    db: Session = Depends(deps.get_db)
):
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="Page and page_size must be greater than 0")

    # Query dasar
    query = db.query(child_model.Child)

    # Tambahkan filter berdasarkan parameter 'search'
    if search:
        query = query.filter(child_model.Child.name.ilike(f"%{search}%"))

    # Hitung total data setelah filter
    total_data = query.count()
    total_page = (total_data + page_size - 1) // page_size  # Hitung total halaman

    # Hitung offset berdasarkan halaman saat ini
    offset = (page - 1) * page_size
    if offset >= total_data > 0:
        raise HTTPException(status_code=404, detail="Page out of range")

    # Ambil data berdasarkan halaman
    items = (query.order_by(child_model.Child.id)
             .offset(offset)
             .limit(page_size)
             .all())

    # Hitung halaman sebelumnya dan berikutnya
    prev_page = page - 1 if page > 1 else 0
    next_page = page + 1 if page < total_page else 0

    return {
        "total_data": total_data,
        "current_page": page,
        "next_page": next_page,
        "prev_page": prev_page,
        "total_page": total_page,
        "items": items,
    }



@router.put("/{child_id}", response_model=child_schema.Child)
def update_child(child_id: int, child: child_schema.ChildUpdate, db: Session = Depends(deps.get_db)):
    # Cari child berdasarkan ID
    db_child = (db.query(child_model.Child)
                .options(joinedload(child_model.Child.images))
                .filter(child_model.Child.id == child_id)
                .first())
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Update atribut Child (selain images)
    for key, value in child.dict(exclude_unset=True).items():  # exclude_unset memastikan hanya nilai yang dikirim diperbarui
        setattr(db_child, key, value)

    db.commit()
    db.refresh(db_child)
    return db_child



@router.delete("/{child_id}", response_model=child_schema.Child)
def delete_child(child_id: int, db: Session = Depends(deps.get_db)):
    db_child = db.query(child_model.Child).options(joinedload(child_model.Child.images)).filter(
        child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Delete associated image files
    for image in db_child.images:
        remove_file(image.image_url)

    # Delete the child (cascade will handle deleting associated images in the database)
    db.delete(db_child)
    db.commit()
    return db_child


@router.post("/{child_id}/image", response_model=child_image_schema.ChildImage)
async def add_child_image(
    child_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(deps.get_db)
):
    # Verifikasi apakah child dengan ID yang diberikan ada
    db_child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Simpan file gambar
    filepath = save_file(image)

    # Buat entri dalam database
    db_image = child_image_model.ChildImage(
        child_id=child_id,
        original_filename=image.filename,
        image_url=filepath
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    # Kembalikan data gambar yang disimpan
    return db_image


@router.delete("/images/{image_id}", response_model=child_image_schema.ChildImage)
def delete_image(image_id: int, db: Session = Depends(deps.get_db)):
    db_image = db.query(child_image_model.ChildImage).filter(child_image_model.ChildImage.id == image_id).first()
    if db_image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove the file from the filesystem
    remove_file(db_image.image_url.lstrip("/"))

    db.delete(db_image)
    db.commit()
    return db_image
