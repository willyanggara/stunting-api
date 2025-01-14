from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+aiomysql://username:password@localhost/stunting_db"
    API_V1_STR: str = "/api"

    class Config:
        env_file = ".env"

settings = Settings()