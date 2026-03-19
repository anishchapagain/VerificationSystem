
import asyncio
from sqlalchemy import select
from backend.db.database import AsyncSessionLocal
from backend.db.models import User

async def list_users():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        print(f"Total Users: {len(users)}")
        for u in users:
            print(f"ID: {u.id}, Email: {u.email}, Name: {u.name}")

if __name__ == "__main__":
    asyncio.run(list_users())
