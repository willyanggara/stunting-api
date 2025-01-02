from sqlalchemy import Column, Integer, Float, DateTime
from app.db.base import Base
from datetime import datetime
class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    mae_height = Column(Float, nullable=False)
    mae_weight = Column(Float, nullable=False)
    rmse_height = Column(Float, nullable=False)
    rmse_weight = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)