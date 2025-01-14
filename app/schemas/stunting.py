from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TrainModelResponse(BaseModel):
    value: bool
    message: str

class CrossValidationResponse(BaseModel):
    k_fold: int
    mae_height_scores: List[float]
    mae_weight_scores: List[float]
    rmse_height_scores: List[float]
    rmse_weight_scores: List[float]
    mean_mae_height: float
    mean_mae_weight: float
    mean_rmse_height: float
    mean_rmse_weight: float

class ModelResponse(BaseModel):
    message: str
    model_exists: bool
    model_path: Optional[str] = None
    model_modified: Optional[str] = None
    scaler_path: Optional[str] = None
    scaler_modified: Optional[str] = None

class PredictionResponse(BaseModel):
    actual_height: float
    actual_weight: float
    actual_stunting: bool
    predicted_height: float
    predicted_weight: float
    predicted_stunting: bool
    predicted_wasting: bool
    predicted_overweight: bool

class PredictAllResponse(BaseModel):
    total_predictions: int
    predictions: List[PredictionResponse]

class ModelMetrics(BaseModel):
    mae_height: float
    mae_weight: float
    rmse_height: float
    rmse_weight: float
    created_at: str

class EvaluationResponse(BaseModel):
    value: bool
    message: str
    metrics: ModelMetrics

class SystemPerformanceResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int