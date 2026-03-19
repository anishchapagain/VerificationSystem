"""
Integration Tests — FastAPI Endpoints
========================================
Uses httpx.AsyncClient with the ASGI transport (no real server needed).
"""
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Health endpoint should return 200 with expected fields."""
    with patch("backend.routers.health.check_db_health", new_callable=AsyncMock, return_value=True):
        resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data
    assert "database" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_list_signatures_empty(async_client, mock_db_session):
    """List signatures returns 200 with empty list for new user."""
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
    with patch("backend.routers.signature.get_db", return_value=mock_db_session):
        with patch("backend.db.crud.SignatureCRUD.get_by_user", new_callable=AsyncMock, return_value=[]):
            resp = await async_client.get("/api/signatures/999")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["signatures"] == []


@pytest.mark.asyncio
async def test_delete_nonexistent_signature(async_client, mock_db_session):
    """Deleting a non-existent signature should return 404."""
    with patch("backend.db.crud.SignatureCRUD.get_by_id", new_callable=AsyncMock, return_value=None):
        with patch("backend.routers.signature.get_db", return_value=mock_db_session):
            resp = await async_client.delete("/api/signatures/99999")
    assert resp.status_code == 404
