"""
Database models and configuration for PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, synonym
from config import settings

Base = declarative_base()


class ChatMessage(Base):
    """Chat history table to store all conversations"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # To group conversations by session
    role = Column(String)                     # 'user' or 'assistant'
    message = Column(Text)
    # Canonical DB column in the updated schema.
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # Backward-compatible alias so existing code using `timestamp` keeps working.
    timestamp = synonym("created_at")


# Create database engine
engine = create_engine(settings.database_url)

# Create all tables
Base.metadata.create_all(bind=engine)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency to get DB session.
    Use this in FastAPI route dependencies.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
