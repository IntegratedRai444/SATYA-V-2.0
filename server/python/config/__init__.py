"""
Configuration Management
Handles environment variables and application configuration with validation.
"""
import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl, PostgresDsn, field_validator, model_validator
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "SatyaAI"
    APP_ENV: str = Field(default="development", env=["ENVIRONMENT", "NODE_ENV"])
    NODE_ENV: str = "development"
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRES_IN: str = Field(default="30d", env="REFRESH_TOKEN_EXPIRES_IN")

    # Server Configuration
    PORT: int = Field(default=8000, env=["PORT", "SERVER_PORT"])
    HOST: str = "0.0.0.0"
    ENVIRONMENT: str = Field(default="development", env=["ENVIRONMENT", "NODE_ENV"])
    NODE_ENV: str = Field(default="development", env="NODE_ENV")

    # API Configuration
    VITE_API_URL: str = "http://localhost:5001/api/v2"  # Match Node.js backend port
    API_BASE_URL: str = "http://localhost:5001"
    VITE_WS_URL: str = "ws://localhost:5001/ws"
    WS_PATH: str = "/ws"
    WS_PORT: int = 5001  # Match Node.js backend port
    CLIENT_PORT: int = 5173  # Vite default port

    # File Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 104857600  # 100MB
    ALLOWED_FILE_TYPES: str = (
        "image/jpeg,image/png,image/webp,video/mp4,video/webm,audio/mpeg,audio/wav"
    )

    # JWT Configuration
    JWT_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRES_IN: str = "24h"

    # API Settings
    API_V1_STR: str = "/api/v1"
    API_V2_STR: str = "/api/v2"

    # CORS settings
    CORS_ALLOW_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://satyaai.app",
            "https://www.satyaai.app"
        ],
        env=["CORS_ALLOW_ORIGINS", "CORS_ORIGIN"],
    )
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        env=["CORS_ALLOW_METHODS", "CORS_METHODS"],
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=[
            "*",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "X-API-Key",
            "X-Request-Id",
            "X-CSRF-Token"
        ],
        env="CORS_ALLOW_HEADERS"
    )
    CORS_EXPOSE_HEADERS: List[str] = Field(
        default=[
            "Content-Length",
            "Content-Type",
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        env="CORS_EXPOSE_HEADERS",
    )
    CORS_MAX_AGE: int = Field(default=86400, env="CORS_MAX_AGE")  # 24 hours
    CORS_CREDENTIALS: bool = Field(default=True, env="CORS_CREDENTIALS")

    # Database
    DATABASE_URL: str
    TEST_DATABASE_URL: Optional[str] = None

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    SUPABASE_JWT_SECRET: Optional[str] = None
    USE_SUPABASE_CLIENT: bool = False

    # Python Server
    PYTHON_SERVER_URL: str = "http://localhost:8000"
    PYTHON_SERVER_PORT: int = 8000
    PYTHON_SERVER_TIMEOUT: int = 300000  # 5 minutes
    PYTHON_HEALTH_CHECK_URL: str = "http://localhost:8000/health"
    
    # Node.js Server (for reference)
    NODE_SERVER_URL: str = "http://localhost:5001"
    NODE_SERVER_PORT: int = 5001

    # Rate Limiting
    RATE_LIMIT_WINDOW_MS: int = 900000  # 15 minutes
    RATE_LIMIT_MAX: int = 100

    # Security
    SECURE_COOKIES: bool = True
    SESSION_COOKIE_NAME: str = "satya_session"
    SESSION_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_AUTH: str = "10/minute"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
        env_prefix=""
    )

    @classmethod
    def _parse_env_var(cls, field_name: str, raw_val: str) -> Any:
        if field_name in [
            "CORS_ALLOW_ORIGINS",
            "CORS_ALLOW_METHODS",
            "CORS_ALLOW_HEADERS",
            "CORS_EXPOSE_HEADERS",
            "CORS_ORIGIN",
            "CORS_METHODS",
        ]:
            return [x.strip() for x in raw_val.split(",") if x.strip()]
        return raw_val

    # CORS origins are handled by CORS_ALLOW_ORIGINS

    @field_validator("APP_ENV", "NODE_ENV", "ENVIRONMENT", mode='before')
    @classmethod
    def validate_app_env(cls, v: Optional[str], info: FieldValidationInfo) -> str:
        if v is None:
            return "development"
        v = v.lower()
        if v not in ["development", "testing", "production"]:
            raise ValueError(
                f"{info.field_name} must be one of: development, testing, production"
            )
        return v

    @field_validator(
        "CORS_ALLOW_ORIGINS",
        "CORS_ALLOW_METHODS",
        "CORS_ALLOW_HEADERS",
        "CORS_EXPOSE_HEADERS",
        mode='before'
    )
    @classmethod
    def parse_comma_separated_list(cls, v: Union[str, List[str], None]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


# Initialize settings
settings = Settings()

# Apply environment-specific overrides
if settings.APP_ENV == "production":
    settings.SECURE_COOKIES = True
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
elif settings.APP_ENV == "testing":
    settings.DEBUG = True
    settings.TESTING = True

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
)

# Validate required settings in production
if settings.APP_ENV == "production":
    required_vars = ["DATABASE_URL", "SECRET_KEY"]
    missing_vars = [var for var in required_vars if not getattr(settings, var, None)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

# Export settings
__all__ = ["settings"]
