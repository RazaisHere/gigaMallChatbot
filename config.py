"""
Configuration settings for the Mall Chatbot API
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database configuration - full PostgreSQL connection URL
    # Format: postgresql://user:password@host:port/database
    database_url: str
    
    # API Keys
    openai_api_key: str  # For OpenAI API
    
    # SendPulse credentials for social media integration
    sendpulse_client_id: str
    sendpulse_client_secret: str
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Create settings instance
settings = Settings()
