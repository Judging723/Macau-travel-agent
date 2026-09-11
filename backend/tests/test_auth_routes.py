from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, decode_access_token
from app.config import Settings
from app.db.session import get_db_session
from app.main import create_app
from app.models import User


def make_user() -> User:
    now = datetime.now(UTC)
    return User(
        id=uuid4(),
        email="alice@example.com",
        hashed_password="stored-hash",
        display_name="Alice",
        is_active=True,
        created_at=now,
        updated_at=now,
    )


def make_client(session: AsyncSession) -> TestClient:
    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_db_session] = override_db_session
    return TestClient(app)


def test_register_creates_user_without_exposing_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None
    user_id = uuid4()
    now = datetime.now(UTC)

    async def fake_refresh(user: User) -> None:
        user.id = user_id
        user.is_active = True
        user.created_at = now
        user.updated_at = now

    session.refresh.side_effect = fake_refresh
    monkeypatch.setattr(
        "app.api.auth_routes.hash_password",
        lambda password: f"hashed:{password}",
    )

    response = make_client(session).post(
        "/api/v1/auth/register",
        json={
            "email": "Alice@Example.com",
            "password": "strong-password",
            "display_name": "Alice",
        },
    )

    assert response.status_code == 201

    created_user = session.add.call_args.args[0]
    assert created_user.email == "alice@example.com"
    assert created_user.hashed_password == "hashed:strong-password"
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(created_user)

    response_data = response.json()
    assert response_data["id"] == str(user_id)
    assert response_data["email"] == "alice@example.com"
    assert "password" not in response_data
    assert "hashed_password" not in response_data


def test_register_rejects_duplicate_email() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = make_user()

    response = make_client(session).post(
        "/api/v1/auth/register",
        json={
            "email": "alice@example.com",
            "password": "strong-password",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Email already registered"}
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


def test_login_returns_token_for_valid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = make_user()
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = user
    monkeypatch.setattr(
        "app.api.auth_routes.verify_password",
        lambda plain_password, hashed_password: (
            plain_password == "strong-password" and hashed_password == "stored-hash"
        ),
    )

    response = make_client(session).post(
        "/api/v1/auth/login",
        json={
            "email": "alice@example.com",
            "password": "strong-password",
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["token_type"] == "bearer"
    assert decode_access_token(response_data["access_token"]) == user.id


def test_login_rejects_unknown_user() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None

    response = make_client(session).post(
        "/api/v1/auth/login",
        json={
            "email": "nobody@example.com",
            "password": "strong-password",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}


def test_me_resolves_user_from_bearer_token() -> None:
    user = make_user()
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = user
    token = create_access_token(user.id)

    response = make_client(session).get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(user.id)
    session.get.assert_awaited_once_with(User, user.id)


def test_me_rejects_invalid_token() -> None:
    session = AsyncMock(spec=AsyncSession)

    response = make_client(session).get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}
    session.get.assert_not_awaited()
