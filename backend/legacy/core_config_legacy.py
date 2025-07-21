from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "SatyaAI API"
    SECRET_KEY: str = "supersecretkey"
    DATABASE_URL: str = "sqlite:///./satyaai.db"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file = ".env"

settings = Settings() 