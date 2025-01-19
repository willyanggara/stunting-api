import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Query
from pydantic import parse_obj_as
from sqlalchemy import func, case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.models import child as child_model
from app.schemas.child import Child, ChildBase, ChildWithImage, PaginatedChildren, SummaryChildren, \
    PredictionChildResponse
from app.utils.file_helper import save_file_async, remove_file_async

router = APIRouter()

@router.get("/list-static-files")
def list_static_files():
    files = os.listdir("/app/static")
    return {"files": files}

@router.get("/list-child-files")
def list_child_files():
    files = os.listdir("/app/static/child_images")
    return {"files": files}

@router.get("/list-child2-files")
def list_child2_files():
    files = os.listdir("static/child_images")
    return {"files": files}

@router.get("/summary", response_model=SummaryChildren)
async def get_summary(db: AsyncSession = Depends(deps.get_db)):
    query = select(
        func.count(child_model.Child.id).label("total_data"),
        func.sum(case((child_model.Child.is_stunting == True, 1), else_=0)).label("total_stunting"),
        func.sum(case((child_model.Child.is_stunting == False, 1), else_=0)).label("total_not_stunting"),
        func.sum(case((child_model.Child.gender == "Male", 1), else_=0)).label("total_male"),
        func.sum(case((child_model.Child.gender == "Female", 1), else_=0)).label("total_female"),
        func.avg(child_model.Child.height).label("average_height"),
        func.avg(child_model.Child.weight).label("average_weight"),
        func.avg(child_model.Child.age).label("average_age"),
    )

    result = await db.execute(query)
    metrics = result.one()

    return {
        "total_data": metrics.total_data or 0,
        "total_stunting": metrics.total_stunting or 0,
        "total_not_stunting": metrics.total_not_stunting or 0,
        "total_male": metrics.total_male or 0,
        "total_female": metrics.total_female or 0,
        "average_height": round(metrics.average_height, 2) if metrics.average_height else 0.0,
        "average_weight": round(metrics.average_weight, 2) if metrics.average_weight else 0.0,
        "average_age": round(metrics.average_age, 2) if metrics.average_age else 0.0,
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
        db: AsyncSession = Depends(deps.get_db)  # Use AsyncSession
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
        filepath = await save_file_async(image_front)
        child_data["image_front_name"] = filepath
        child_data["image_front_original_name"] = image_front.filename

    if image_back:
        filepath = await save_file_async(image_back)
        child_data["image_back_name"] = filepath
        child_data["image_back_original_name"] = image_back.filename

    if image_left:
        filepath = await save_file_async(image_left)
        child_data["image_left_name"] = filepath
        child_data["image_left_original_name"] = image_left.filename

    if image_right:
        filepath = await save_file_async(image_right)
        child_data["image_right_name"] = filepath
        child_data["image_right_original_name"] = image_right.filename

    child = parse_obj_as(ChildWithImage, child_data)
    db_child = child_model.Child(**child.dict())
    db.add(db_child)
    await db.commit()  # Use async commit
    await db.refresh(db_child)  # Use async refresh
    return db_child


@router.get("/{child_id}", response_model=Child)
async def get_child(child_id: int, request: Request, db: AsyncSession = Depends(deps.get_db)):
    db_child = await db.execute(select(child_model.Child).filter(child_model.Child.id == child_id))
    db_child = db_child.scalars().first()
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
async def list_children(
        page: int = 1,
        page_size: int = 10,
        search: str = Query(None, description="Search for child name"),
        db: AsyncSession = Depends(deps.get_db)
):
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="Page and page_size must be greater than 0")

    query = select(child_model.Child)

    if search:
        query = query.filter(child_model.Child.name.ilike(f"%{search}%"))

    query = query.order_by(child_model.Child.id.desc())
    result = await db.execute(query)
    children = result.scalars().all()

    total_data = len(children)
    total_page = (total_data + page_size - 1) // page_size

    offset = (page - 1) * page_size
    if offset >= total_data > 0:
        raise HTTPException(status_code=404, detail="Page out of range")

    items = children[offset:offset + page_size]

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
async def update_child(child_id: int, child: ChildBase, db: AsyncSession = Depends(deps.get_db)):
    db_child = await db.execute(select(child_model.Child).filter(child_model.Child.id == child_id))
    db_child = db_child.scalars().first()

    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    for key, value in child.dict(exclude_unset=True).items():
        setattr(db_child, key, value)

    await db.commit()
    await db.refresh(db_child)

    return db_child


@router.delete("/{child_id}", response_model=Child)
async def delete_child(child_id: int, db: AsyncSession = Depends(deps.get_db)):
    # Use async query to fetch the child
    db_child = await db.execute(select(child_model.Child).filter(child_model.Child.id == child_id))
    db_child = db_child.scalars().first()

    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Delete associated image files asynchronously
    image_removal_tasks = []
    if db_child.image_front_name:
        image_removal_tasks.append(remove_file_async(str(db_child.image_front_name)))
    if db_child.image_back_name:
        image_removal_tasks.append(remove_file_async(str(db_child.image_back_name)))
    if db_child.image_left_name:
        image_removal_tasks.append(remove_file_async(str(db_child.image_left_name)))
    if db_child.image_right_name:
        image_removal_tasks.append(remove_file_async(str(db_child.image_right_name)))

    # Run all image removal tasks concurrently
    if image_removal_tasks:
        await asyncio.gather(*image_removal_tasks)

    # Delete the child record asynchronously
    await db.delete(db_child)
    await db.commit()

    return db_child


@router.post("/{child_id}/image/{image_type}", response_model=Child)
async def add_child_image(
        child_id: int,
        image_type: str,
        image: UploadFile = File(...),
        db: AsyncSession = Depends(deps.get_db)
):
    db_child = await get_child_and_validate_image_type(child_id, image_type, db)

    # Asynchronously save the image
    image_name = getattr(db_child, f"image_{image_type}_name")
    filepath = await save_file_async(image)

    setattr(db_child, f"image_{image_type}_name", filepath)
    setattr(db_child, f"image_{image_type}_original_name", image.filename)

    # If an image already exists, schedule its removal asynchronously
    if image_name:
        # Use asyncio.create_task to remove the file asynchronously without blocking
        asyncio.create_task(remove_file_async(image_name))

    await db.commit()
    await db.refresh(db_child)
    return db_child


@router.delete("/{child_id}/image/{image_type}", response_model=Child)
async def delete_image(child_id: int, image_type: str, db: AsyncSession = Depends(deps.get_db)):
    db_child = await get_child_and_validate_image_type(child_id, image_type, db)

    image_name = getattr(db_child, f"image_{image_type}_name")
    if image_name:
        # Use asyncio.create_task to remove the file asynchronously without blocking
        asyncio.create_task(remove_file_async(image_name))
        setattr(db_child, f"image_{image_type}_name", None)
        setattr(db_child, f"image_{image_type}_original_name", None)
        await db.commit()
        await db.refresh(db_child)

    return db_child


async def get_child_and_validate_image_type(child_id: int, image_type: str, db: AsyncSession) -> child_model.Child:
    # Use async select to query the child in the database
    result = await db.execute(select(child_model.Child).filter(child_model.Child.id == child_id))
    db_child = result.scalars().first()

    if db_child is None:
        raise HTTPException(status_code=404, detail="Child not found")

    if image_type not in ["front", "back", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    return db_child
