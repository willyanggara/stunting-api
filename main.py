import os
import logging
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
logger = logging.getLogger(__name__)
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

# if __name__ == "__main__":
#     # Get the host and port from environment variables, default to 0.0.0.0 and 8000
#     host = os.getenv("HOST", "0.0.0.0")  # If the environment variable is not set, use "0.0.0.0"
#     port = int(os.getenv("PORT", 8080))  # Default to 8080, which is commonly used in Railway deployments
#
#     # Log the details
#     logger.info(f"Starting application at http://{host}:{port}")
#
#     # Run Uvicorn
#     uvicorn.run(app, host=host, port=port, log_level="info")