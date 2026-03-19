import asyncio
from backend.db.database import AsyncSessionLocal, engine
from backend.db.models import User
from backend.services.auth import AuthService
from sqlalchemy import select

async def seed_user():
    print("Seeding default user...")
    async with AsyncSessionLocal() as session:
        # Check if user with ID 1 already exists
        result = await session.execute(select(User).where(User.id == 1))
        user = result.scalar_one_or_none()
        
        if user:
            print(f"User already exists: {user}")
            return

        # Create default user
        hashed_pw = AuthService.hash_password("password123")
        new_user = User(
            id=1,
            name="Default User",
            email="user@example.com",
            hashed_password=hashed_pw
        )
        
        session.add(new_user)
        try:
            await session.commit()
            print("Default user (ID: 1) created successfully.")
        except Exception as e:
            await session.rollback()
            print(f"Failed to seed user: {e}")

if __name__ == "__main__":
    asyncio.run(seed_user())
