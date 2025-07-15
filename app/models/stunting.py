from sqlalchemy import JSON, Column, Integer, Float, DateTime
from app.db.base import Base
from datetime import datetime
class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    mae_height = Column(Float, nullable=False)
    mae_weight = Column(Float, nullable=False)
    rmse_height = Column(Float, nullable=False)
    rmse_weight = Column(Float, nullable=False)
    # config_used = Column(JSON)  # Untuk menyimpan konfigurasi evaluasi
    created_at = Column(DateTime, default=datetime.utcnow)