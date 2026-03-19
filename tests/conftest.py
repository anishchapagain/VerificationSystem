"""
Pytest Configuration & Shared Fixtures
=========================================
Module  : tests/conftest.py
"""
import io
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def sample_embedding():
    """Return a random L2-normalised 512-D embedding vector."""
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def sample_image_array():
    """Return a 128×256 float32 array simulating a preprocessed signature."""
    return np.random.rand(128, 256).astype(np.float32)


@pytest.fixture
def mock_db_session():
    """Return a mocked AsyncSession."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest_asyncio.fixture
async def async_client():
    """Return an httpx AsyncClient wired to the FastAPI app."""
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
