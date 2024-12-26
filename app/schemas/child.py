from pydantic import BaseModel, Field
from typing import List, Optional
from .child_image import ChildImage, ChildImageCreate
from .generic_response import GenericResponse


class ChildBase(BaseModel):
    id: Optional[int] = None  # id can be null
    name: str = Field(..., min_length=1)
    gender: str = Field(..., pattern="^(Male|Female)$")
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    age: int = Field(..., ge=0)
    is_stunting: bool

class ChildCreate(ChildBase):
    images: List[ChildImageCreate] = Field(default_factory=list)

class ChildUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    gender: Optional[str] = Field(None, pattern="^(Male|Female)$")
    height: Optional[float] = Field(None, gt=0)
    weight: Optional[float] = Field(None, gt=0)
    age: Optional[int] = Field(None, ge=0)
    is_stunting: Optional[bool] = None

class Child(ChildBase):
    id: int
    images: List[ChildImage] = Field(default_factory=list)

    class Config:
        from_attributes = True  # new name for orm_mode in Pydantic v2

class PaginatedChildren(BaseModel):
    total_data: int = Field(..., ge=0)
    current_page: int = Field(..., ge=1)
    next_page: int = Field(..., ge=0)
    prev_page: int = Field(..., ge=0)
    total_page: int = Field(..., ge=0)
    items: List[ChildBase] = Field(default_factory=list)

class ChildResponse(GenericResponse[Child]):
    pass

class PaginatedChildrenResponse(GenericResponse[PaginatedChildren]):
    pass


class SummaryChildren(BaseModel):
    total_data: int = Field(..., ge=0)
    total_stunting: int = Field(..., ge=0)  # Allow 0 if no stunting data exists
    total_not_stunting: int = Field(..., ge=0)
    total_male: int = Field(..., ge=0)
    total_female: int = Field(..., ge=0)
    average_height: Optional[float] = Field(None, ge=0)  # Allow None for no data
    average_weight: Optional[float] = Field(None, ge=0)  # Allow None for no data
    average_age: Optional[float] = Field(None, ge=0)     # Allow None for no data