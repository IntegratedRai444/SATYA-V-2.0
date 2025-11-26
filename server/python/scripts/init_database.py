"""
Database Initialization Script
Creates all database tables and optionally seeds initial data
"""

import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
python_dir = current_dir.parent
sys.path.insert(0, str(python_dir))

from services.database import DatabaseManager
from database.base import Base, engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize database tables"""
    logger.info("=" * 60)
    logger.info("🗄️  Database Initialization")
    logger.info("=" * 60)
    
    try:
        # Create database manager
        db_manager = DatabaseManager()
        
        # Create all tables
        logger.info("Creating database tables...")
        db_manager.create_tables()
        
        logger.info("=" * 60)
        logger.info("✅ Database initialization complete!")
        logger.info("=" * 60)
        
        # Print table info
        logger.info("\nCreated tables:")
        for table in Base.metadata.sorted_tables:
            logger.info(f"  - {table.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def seed_test_data():
    """Seed database with test data (optional)"""
    logger.info("\n" + "=" * 60)
    logger.info("🌱 Seeding Test Data")
    logger.info("=" * 60)
    
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        db_manager = DatabaseManager()
        
        # Create test user
        test_user = db_manager.create_user(
            username="testuser",
            email="test@satyaai.com",
            hashed_password=pwd_context.hash("testpassword123"),
            full_name="Test User"
        )
        
        if test_user:
            logger.info(f"✅ Test user created: {test_user.username}")
        
        logger.info("=" * 60)
        logger.info("✅ Test data seeded successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to seed test data: {e}")
        return False


if __name__ == "__main__":
    # Initialize database
    success = init_database()
    
    # Optionally seed test data
    if success and "--seed" in sys.argv:
        seed_test_data()
