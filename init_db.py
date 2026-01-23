"""
Database initialization script
Creates tables for chat messages
"""

from database import Base, engine


def init_database():
    """Create database tables"""
    try:
        # Create all tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
        print("✅ Database is ready to use!")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise


if __name__ == "__main__":
    init_database()
