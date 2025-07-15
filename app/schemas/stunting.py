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
    training_stats: Optional[dict] = None

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
    acuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
        
class TrainingConfig(BaseModel):
    use_front: bool = True
    use_back: bool = True
    use_left: bool = True
    use_right: bool = True
    use_all: bool = False
    optimizer: str = "adam"  # "adam" or "sgd"
    learning_rate: float = 0.0001
    batch_size: int = 16
    epochs: int = 100
    test_size: float = 0.1
    
class EvaluationConfig(BaseModel):
    use_front: bool = True
    use_back: bool = True
    use_left: bool = True
    use_right: bool = True
    use_all: bool = False  # Jika True, abaikan yang lain dan gunakan semua
    
class PredictionConfig(BaseModel):
    use_front: bool = True
    use_back: bool = True 
    use_left: bool = True
    use_right: bool = True
    use_all: bool = False