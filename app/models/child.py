from sqlalchemy import Column, Integer, String, Float, Boolean
from app.db.base import Base

class Child(Base):
    __tablename__ = "child"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=False)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    age = Column(Integer, nullable=True)
    is_stunting = Column(Boolean, nullable=False)
    image_front_name = Column(String(500), nullable=True)
    image_front_original_name = Column(String(255), nullable=True)
    image_back_name = Column(String(500), nullable=True)
    image_back_original_name = Column(String(255), nullable=True)
    image_left_name = Column(String(500), nullable=True)
    image_left_original_name = Column(String(255), nullable=True)
    image_right_name = Column(String(500), nullable=True)
    image_right_original_name = Column(String(255), nullable=True)
    predict_height = Column(Float, nullable=True)
    predict_weight = Column(Float, nullable=True)
    predict_stunting = Column(Boolean, nullable=True)
    predict_wasting = Column(Boolean, nullable=True)
    predict_overweight = Column(Boolean, nullable=True)

