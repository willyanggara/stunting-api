from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+aiomysql://username:password@localhost/stunting_db"
    API_V1_STR: str = "/api"
    TF_CPP_MIN_LOG_LEVEL: str = "2"
    TF_ENABLE_ONEDNN_OPTS: str = "0"

    class Config:
        env_file = ".env"

settings = Settings()