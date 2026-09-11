from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth_routes import get_current_user
from app.config import Settings
from app.db.session import get_db_session
from app.main import create_app
from app.models import User


def make_client(user: User, session: AsyncSession) -> TestClient:
    async def override_current_user() -> User:
        return user

    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db_session] = override_db_session
    return TestClient(app)


def test_chat_creates_conversation_and_calls_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)
    answer = AsyncMock(return_value="澳门博物馆适合文化旅行。")
    orchestrator = SimpleNamespace(answer=answer)
    monkeypatch.setattr(
        "app.api.chat_routes.get_travel_orchestrator",
        lambda: orchestrator,
    )

    response = make_client(user, session).post(
        "/api/v1/chat",
        json={"message": "推荐澳门的文化景点"},
    )

    assert response.status_code == 200
    response_data = response.json()
    conversation_id = UUID(response_data["conversation_id"])
    assert response_data["reply"] == "澳门博物馆适合文化旅行。"
    answer.assert_awaited_once_with(
        message="推荐澳门的文化景点",
        user_id=user.id,
        conversation_id=conversation_id,
        session=session,
    )


def test_chat_preserves_supplied_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    answer = AsyncMock(return_value="记得您的旅行偏好。")
    monkeypatch.setattr(
        "app.api.chat_routes.get_travel_orchestrator",
        lambda: SimpleNamespace(answer=answer),
    )

    response = make_client(user, session).post(
        "/api/v1/chat",
        json={
            "message": "我之前说过什么？",
            "conversation_id": str(conversation_id),
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == str(conversation_id)
    assert answer.await_args.kwargs["conversation_id"] == conversation_id


def test_chat_returns_503_when_configuration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)

    def raise_configuration_error() -> None:
        raise ValueError("DeepSeek API key is not configured")

    monkeypatch.setattr(
        "app.api.chat_routes.get_travel_orchestrator",
        raise_configuration_error,
    )

    response = make_client(user, session).post(
        "/api/v1/chat",
        json={"message": "推荐澳门景点"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "DeepSeek API key is not configured",
    }


def test_chat_returns_502_when_agent_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)
    answer = AsyncMock(side_effect=RuntimeError("model failed"))
    monkeypatch.setattr(
        "app.api.chat_routes.get_travel_orchestrator",
        lambda: SimpleNamespace(answer=answer),
    )

    response = make_client(user, session).post(
        "/api/v1/chat",
        json={"message": "推荐澳门景点"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Travel agent service is unavailable",
    }
