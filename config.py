"""
Configuration settings for the Mall Chatbot API
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database configuration - can use either DATABASE_URL or individual components
    database_url: Optional[str] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_host: str = "localhost"
    db_port: str = "5432"
    
    # API Keys
    openai_api_key: str  # For OpenAI API
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Create settings instance
_settings = Settings()

# Build database_url if individual components are provided
if not _settings.database_url:
    if _settings.db_name and _settings.db_user and _settings.db_password:
        _settings.database_url = f"postgresql://{_settings.db_user}:{_settings.db_password}@{_settings.db_host}:{_settings.db_port}/{_settings.db_name}"
    elif not _settings.database_url:
        raise ValueError(
            "Either DATABASE_URL must be provided in .env file, "
            "or all of db_name, db_user, and db_password must be provided"
        )

settings = _settings
