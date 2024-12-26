from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import relationship

from app.db.base import Base

class Child(Base):
    __tablename__ = "child"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=False)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    is_stunting = Column(Boolean, nullable=False)
    images = relationship("ChildImage", back_populates="child", cascade="all, delete-orphan")
