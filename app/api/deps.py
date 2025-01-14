from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import SessionLocal

async def get_db() -> AsyncGenerator:
    async with SessionLocal() as db:
        yield db
