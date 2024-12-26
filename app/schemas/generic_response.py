from pydantic import BaseModel
from typing import Optional, Generic, TypeVar

T = TypeVar('T')

class GenericResponse(BaseModel, Generic[T]):
    StatusCode: int
    StatusMessage: str
    Value: Optional[T] = None

