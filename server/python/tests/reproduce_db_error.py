import os
import sys
from datetime import datetime
from pathlib import Path

# Add server/python to path
current_dir = Path(__file__).parent
server_dir = current_dir.parent
sys.path.append(str(server_dir))

from services.database import DatabaseManager


def test_create_user():
    print("Testing create_user...")
    db = DatabaseManager()

    # Ensure tables exist
    db.create_tables()

    username = f"test_user_{int(datetime.utcnow().timestamp())}"
    email = f"{username}@example.com"
    password = "password123"

    try:
        user = db.create_user(username, email, password, "Test User")
        if user:
            print(f"✅ User created: {user.username}")
        else:
            print("❌ Failed to create user (returned None)")
    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    test_create_user()
