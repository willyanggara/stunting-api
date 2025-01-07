from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Query
from pydantic import parse_obj_as
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api import deps
from app.models import child as child_model
from app.schemas.child import Child, ChildBase, ChildWithImage, PaginatedChildren, SummaryChildren, \
    PredictionChildResponse
from app.utils.file_helper import save_file, remove_file

router = APIRouter()

@router.get("/summary", response_model=SummaryChildren)
def get_summary(db: Session = Depends(deps.get_db)):
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

    total_stunting = db.query(func.count()).filter(child_model.Child.is_stunting == True).scalar()
    total_not_stunting = db.query(func.count()).filter(child_model.Child.is_stunting == False).scalar()
    total_male = db.query(func.count()).filter(child_model.Child.gender == "Male").scalar()
    total_female = db.query(func.count()).filter(child_model.Child.gender == "Female").scalar()

    average_height = db.query(func.avg(child_model.Child.height)).scalar()
    average_weight = db.query(func.avg(child_model.Child.weight)).scalar()
    average_age = db.query(func.avg(child_model.Child.age)).scalar()

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


@router.post("", response_model=Child)
async def create_child(
    name: str = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    age: Optional[int] = Form(None),
    is_stunting: bool = Form(...),
    image_front: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    image_left: Optional[UploadFile] = File(None),
    image_right: Optional[UploadFile] = File(None),
    db: Session = Depends(deps.get_db)
):
    child_data = {
        "name": name,
        "gender": gender,
        "height": height,
        "weight": weight,
        "age": age if age is not None else None,
        "is_stunting": is_stunting
    }

    if image_front:
        filepath = save_file(image_front)
        child_data["image_front_name"] = filepath
        child_data["image_front_original_name"] = image_front.filename

    if image_back:
        filepath = save_file(image_back)
        child_data["image_back_name"] = filepath
        child_data["image_back_original_name"] = image_back.filename

    if image_left:
        filepath = save_file(image_left)
        child_data["image_left_name"] = filepath
        child_data["image_left_original_name"] = image_left.filename

    if image_right:
        filepath = save_file(image_right)
        child_data["image_right_name"] = filepath
        child_data["image_right_original_name"] = image_right.filename

    child = parse_obj_as(ChildWithImage, child_data)
    db_child = child_model.Child(**child.dict())
    db.add(db_child)
    db.commit()
    db.refresh(db_child)
    return db_child


@router.get("/{child_id}", response_model=Child)
def get_child(child_id: int, request: Request, db: Session = Depends(deps.get_db)):
    db_child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    base_url = str(request.base_url)
    if db_child.image_front_name:
        db_child.image_front_name = f"{base_url}static/child_images/{db_child.image_front_name}"
    if db_child.image_back_name:
        db_child.image_back_name = f"{base_url}static/child_images/{db_child.image_back_name}"
    if db_child.image_left_name:
        db_child.image_left_name = f"{base_url}static/child_images/{db_child.image_left_name}"
    if db_child.image_right_name:
        db_child.image_right_name = f"{base_url}static/child_images/{db_child.image_right_name}"

    child_data = Child.from_orm(db_child)
    # Include prediction data if available
    if db_child.predict_height is not None and db_child.predict_weight is not None:
        child_data.prediction = PredictionChildResponse(
            actual_height=float(db_child.height or 0),
            actual_weight=float(db_child.weight or 0),
            actual_stunting=bool(db_child.is_stunting or False),
            predicted_height=float(db_child.predict_height or 0),
            predicted_weight=float(db_child.predict_weight or 0),
            predicted_stunting=bool(db_child.predict_stunting or False),
            predicted_wasting=bool(db_child.predict_wasting or False),
            predicted_overweight=bool(db_child.predict_overweight or False)
        )

    return child_data


@router.get("", response_model=PaginatedChildren)
def list_children(
        page: int = 1,
        page_size: int = 10,
        search: str = Query(None, description="Search for child name"),
        db: Session = Depends(deps.get_db)
):
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="Page and page_size must be greater than 0")

    query = db.query(child_model.Child)

    if search:
        query = query.filter(child_model.Child.name.ilike(f"%{search}%"))

    total_data = query.count()
    total_page = (total_data + page_size - 1) // page_size

    offset = (page - 1) * page_size
    if offset >= total_data > 0:
        raise HTTPException(status_code=404, detail="Page out of range")

    items = (query.order_by(child_model.Child.id)
             .offset(offset)
             .limit(page_size)
             .all())

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


@router.put("/{child_id}", response_model=Child)
def update_child(child_id: int, child: ChildBase, db: Session = Depends(deps.get_db)):
    db_child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    for key, value in child.dict(exclude_unset=True).items():
        setattr(db_child, key, value)

    db.commit()
    db.refresh(db_child)
    return db_child


@router.delete("/{child_id}", response_model=Child)
def delete_child(child_id: int, db: Session = Depends(deps.get_db)):
    db_child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Delete associated image files
    if db_child.image_front_name:
        remove_file(str(db_child.image_front_name))
    if db_child.image_back_name:
        remove_file(str(db_child.image_back_name))
    if db_child.image_left_name:
        remove_file(str(db_child.image_left_name))
    if db_child.image_right_name:
        remove_file(str(db_child.image_right_name))

    db.delete(db_child)
    db.commit()
    return db_child


@router.post("/{child_id}/image/{image_type}", response_model=Child)
async def add_child_image(
        child_id: int,
        image_type: str,
        image: UploadFile = File(...),
        db: Session = Depends(deps.get_db)
):
    db_child = get_child_and_validate_image_type(child_id, image_type, db)

    image_name = getattr(db_child, f"image_{image_type}_name")
    filepath = save_file(image)

    setattr(db_child, f"image_{image_type}_name", filepath)
    setattr(db_child, f"image_{image_type}_original_name", image.filename)

    if image_name:
        remove_file(image_name)

    db.commit()
    db.refresh(db_child)
    return db_child


@router.delete("/{child_id}/image/{image_type}", response_model=Child)
def delete_image(child_id: int, image_type: str, db: Session = Depends(deps.get_db)):
    db_child = get_child_and_validate_image_type(child_id, image_type, db)

    image_name = getattr(db_child, f"image_{image_type}_name")
    if image_name:
        remove_file(image_name)
        setattr(db_child, f"image_{image_type}_name", None)
        setattr(db_child, f"image_{image_type}_original_name", None)
        db.commit()
        db.refresh(db_child)

    return db_child

def get_child_and_validate_image_type(child_id: int, image_type: str, db: Session) -> child_model.Child:
    db_child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    if image_type not in ["front", "back", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    return db_child

