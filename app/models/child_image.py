from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base

class ChildImage(Base):
    __tablename__ = "child_images"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child.id", ondelete="CASCADE"), nullable=False)
    image_url = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=False)
    child = relationship("Child", back_populates="images")