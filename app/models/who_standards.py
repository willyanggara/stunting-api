from sqlalchemy import Column, Integer, Float, String
from app.db.base import Base

class HeightForAge(Base):
    __tablename__ = "height_for_age"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer)
    months = Column(Integer)
    sd_3_negative = Column(Float)
    sd_2_negative = Column(Float)
    sd_1_negative = Column(Float)
    median = Column(Float)
    sd_1_positive = Column(Float)
    sd_2_positive = Column(Float)
    sd_3_positive = Column(Float)
    gender = Column(String(10))

class WeightForAge(Base):
    __tablename__ = "weight_for_age"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer)
    months = Column(Integer)
    sd_3_negative = Column(Float)
    sd_2_negative = Column(Float)
    sd_1_negative = Column(Float)
    median = Column(Float)
    sd_1_positive = Column(Float)
    sd_2_positive = Column(Float)
    sd_3_positive = Column(Float)
    gender = Column(String(10))

