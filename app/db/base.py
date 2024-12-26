from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Impor semua model agar Base.metadata dapat mengenali mereka
#from app.models.child import Child
#from app.models.child_image import ChildImage
import app.models