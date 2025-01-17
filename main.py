import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import child
from app.core.config import settings
from app.api.endpoints import stunting

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = settings.TF_CPP_MIN_LOG_LEVEL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = settings.TF_ENABLE_ONEDNN_OPTS
app = FastAPI(title="Stunting Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(child.router, prefix=f"{settings.API_V1_STR}/children", tags=["children"])
app.include_router(stunting.router, prefix=f"{settings.API_V1_STR}/stunting", tags=["stunting"])

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 if not set
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    uvicorn.run(app, host=host, port=port)