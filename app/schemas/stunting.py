from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TrainModelResponse(BaseModel):
    value: bool
    message: str

class CheckModelResponse(BaseModel):
    model_exists: bool
    model_path: Optional[str] = None
    model_modified: Optional[datetime] = None
    scaler_path: Optional[str] = None
    scaler_modified: Optional[datetime] = None

class PredictionResponse(BaseModel):
    child_id: int
    actual_height: float
    actual_weight: float
    predicted_height: float
    predicted_weight: float

class ModelMetrics(BaseModel):
    mae_height: float
    mae_weight: float
    rmse_height: float
    rmse_weight: float
    created_at: datetime

class EvaluationResponse(BaseModel):
    value: bool
    message: str
    metrics: ModelMetrics