from pydantic import BaseModel, Field
from typing import Optional
from .generic_response import GenericResponse

class ChildBase(BaseModel):
    id: Optional[int] = None
    name: str = Field(..., min_length=1)
    gender: str = Field(..., pattern="^(Male|Female)$")
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    age: Optional[int] = Field(None, ge=0)
    is_stunting: bool

class ChildWithImage(ChildBase):
    image_front_name: Optional[str] = None
    image_front_original_name: Optional[str] = None
    image_back_name: Optional[str] = None
    image_back_original_name: Optional[str] = None
    image_left_name: Optional[str] = None
    image_left_original_name: Optional[str] = None
    image_right_name: Optional[str] = None
    image_right_original_name: Optional[str] = None

class Child(ChildWithImage):
    id: int

    class Config:
        from_attributes = True

class PaginatedChildren(BaseModel):
    total_data: int = Field(..., ge=0)
    current_page: int = Field(..., ge=1)
    next_page: int = Field(..., ge=0)
    prev_page: int = Field(..., ge=0)
    total_page: int = Field(..., ge=0)
    items: list[ChildBase] = Field(default_factory=list)

class SummaryChildren(BaseModel):
    total_data: int = Field(..., ge=0)
    total_stunting: int = Field(..., ge=0)
    total_not_stunting: int = Field(..., ge=0)
    total_male: int = Field(..., ge=0)
    total_female: int = Field(..., ge=0)
    average_height: Optional[float] = Field(None, ge=0)
    average_weight: Optional[float] = Field(None, ge=0)
    average_age: Optional[float] = Field(None, ge=0)

