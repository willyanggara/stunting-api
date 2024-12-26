from pydantic import BaseModel
from typing import List

from app.schemas.generic_response import GenericResponse


class ChildImageBase(BaseModel):
    image_url: str
    original_filename: str

class ChildImageCreate(ChildImageBase):
    pass

class ChildImageEdit(ChildImageBase):
    id: int

    class Config:
        from_attributes = True

class ChildImage(ChildImageBase):
    id: int
    child_id: int

    class Config:
        from_attributes = True

class ChildImageResponse(GenericResponse[List[ChildImage]]):
    pass