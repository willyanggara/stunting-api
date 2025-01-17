from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import child
from app.core.config import settings
from app.api.endpoints import stunting # Import stunting router

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