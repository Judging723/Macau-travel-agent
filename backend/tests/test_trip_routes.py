from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.trip_agent import ReflectedPlan
from app.api.auth_routes import get_current_user
from app.api.trip_routes import apply_trip_dates, build_budget_analysis
from app.auth import create_access_token
from app.config import TARGET_CITY, TARGET_COUNTRY, Settings
from app.db.session import get_db_session
from app.main import create_app
from app.models import Trip, User
from app.schemas import GeneratedItinerary, PlanReflection


def make_trip(user_id: object) -> Trip:
    now = datetime.now(UTC)
    return Trip(
        id=uuid4(),
        user_id=user_id,
        title="澳门文化之旅",
        destination=TARGET_CITY,
        country=TARGET_COUNTRY,
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 1),
        budget=1000,
        num_people=1,
        status="draft",
        itinerary=[],
        preferences={"interests": ["博物馆"]},
        created_at=now,
        updated_at=now,
    )


def make_authenticated_client(
    current_user: User,
    session: AsyncSession,
) -> TestClient:
    async def override_current_user() -> User:
        return current_user

    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db_session] = override_db_session
    return TestClient(app)


def test_list_trips_requires_authentication() -> None:
    settings = Settings(
        environment="test",
        api_prefix="/api/v1",
    )
    app = create_app(settings)

    response = TestClient(app).get("/api/v1/trips")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated",
    }
    assert response.headers["www-authenticate"] == "Bearer"


def test_list_trips_returns_current_users_macau_trips() -> None:
    user_id = uuid4()
    current_user = User(
        id=user_id,
        email="alice@example.com",
        hashed_password="test-hash",
        display_name="Alice",
        is_active=True,
    )

    now = datetime.now(UTC)
    trip = Trip(
        id=uuid4(),
        user_id=user_id,
        title="澳门文化之旅",
        destination=TARGET_CITY,
        country=TARGET_COUNTRY,
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 3),
        budget=3000,
        num_people=2,
        status="draft",
        itinerary=[],
        preferences={"interests": ["历史文化"]},
        created_at=now,
        updated_at=now,
    )

    session = AsyncMock()
    session.scalars.return_value = [trip]

    async def override_current_user() -> User:
        return current_user

    async def override_db_session() -> AsyncMock:
        return session

    app = create_app(
        Settings(
            environment="test",
            api_prefix="/api/v1",
        )
    )
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db_session] = override_db_session

    response = TestClient(app).get("/api/v1/trips")

    assert response.status_code == 200

    response_data = response.json()

    assert len(response_data) == 1
    assert response_data[0]["user_id"] == str(user_id)
    assert response_data[0]["title"] == "澳门文化之旅"
    assert response_data[0]["destination"] == TARGET_CITY
    assert response_data[0]["budget"] == 3000

    statement = session.scalars.await_args.args[0]
    parameters = statement.compile().params

    assert user_id in parameters.values()
    assert TARGET_CITY in parameters.values()


def test_list_trips_accepts_a_real_bearer_token() -> None:
    user = User(id=uuid4(), is_active=True)
    trip = make_trip(user.id)
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = user
    session.scalars.return_value = [trip]

    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_db_session] = override_db_session
    token = create_access_token(user.id)

    response = TestClient(app).get(
        "/api/v1/trips",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()[0]["user_id"] == str(user.id)
    session.get.assert_awaited_once_with(User, user.id)
    session.scalars.assert_awaited_once()


def test_create_trip_assigns_current_user_and_macau() -> None:
    user_id = uuid4()
    trip_id = uuid4()
    now = datetime.now(UTC)

    current_user = User(
        id=user_id,
        email="alice@example.com",
        hashed_password="test-hash",
        display_name="Alice",
        is_active=True,
    )

    session = AsyncMock(spec=AsyncSession)

    async def fake_refresh(trip: Trip) -> None:
        trip.id = trip_id
        trip.status = "draft"
        trip.itinerary = []
        trip.created_at = now
        trip.updated_at = now

    session.refresh.side_effect = fake_refresh

    async def override_current_user() -> User:
        return current_user

    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(
        Settings(
            environment="test",
            api_prefix="/api/v1",
        )
    )
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db_session] = override_db_session

    response = TestClient(app).post(
        "/api/v1/trips",
        json={
            "title": "澳门三日文化之旅",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "budget": 3000,
            "num_people": 2,
            "preferences": {
                "interests": ["历史文化"],
            },
        },
    )

    assert response.status_code == 201

    created_trip = session.add.call_args.args[0]

    assert created_trip.user_id == user_id
    assert created_trip.destination == TARGET_CITY
    assert created_trip.country == TARGET_COUNTRY
    assert created_trip.title == "澳门三日文化之旅"

    session.add.assert_called_once_with(created_trip)
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(created_trip)

    response_data = response.json()

    assert response_data["id"] == str(trip_id)
    assert response_data["user_id"] == str(user_id)
    assert response_data["destination"] == TARGET_CITY
    assert response_data["country"] == TARGET_COUNTRY
    assert response_data["status"] == "draft"


def test_generate_trip_plan_saves_itinerary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = uuid4()
    now = datetime.now(UTC)

    current_user = User(
        id=user_id,
        email="alice@example.com",
        hashed_password="test-hash",
        display_name="Alice",
        is_active=True,
    )

    trip = Trip(
        id=uuid4(),
        user_id=user_id,
        title="澳门文化之旅",
        destination=TARGET_CITY,
        country=TARGET_COUNTRY,
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 1),
        budget=1000,
        num_people=1,
        status="draft",
        itinerary=[],
        preferences={"interests": ["博物馆"]},
        created_at=now,
        updated_at=now,
    )

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = trip

    generated_plan = GeneratedItinerary(
        summary="澳门文化一日游",
        days=[
            {
                "day_number": 1,
                "travel_date": "2026-10-01",
                "title": "澳门历史城区",
                "items": [
                    {
                        "time": "09:00",
                        "type": "activity",
                        "description": "参观澳门博物馆",
                        "cost": 100,
                    },
                    {
                        "time": "12:00",
                        "type": "food",
                        "description": "品尝澳门美食",
                        "cost": 150,
                    },
                ],
            }
        ],
    )

    reflection = PlanReflection(
        score=9,
        feedback="计划合理",
        issues=[],
    )

    plan_mock = AsyncMock(
        return_value=ReflectedPlan(
            plan=generated_plan,
            reflection=reflection,
            revised=False,
        )
    )
    knowledge_search_mock = AsyncMock(
        return_value=[
            {
                "title": "澳门旅游指南",
                "content": "澳门博物馆资料",
            }
        ]
    )
    web_search_mock = AsyncMock(
        return_value=[
            {
                "title": "澳门实时资料",
                "url": "https://example.com/macau",
                "content": "景点开放信息",
            }
        ]
    )

    orchestrator = SimpleNamespace(
        search_web=web_search_mock,
        trip_agent=SimpleNamespace(plan=plan_mock),
    )

    monkeypatch.setattr(
        "app.api.trip_routes.get_travel_orchestrator",
        lambda: orchestrator,
    )
    monkeypatch.setattr(
        "app.api.trip_routes.search_reranked_knowledge",
        knowledge_search_mock,
    )

    async def override_current_user() -> User:
        return current_user

    async def override_db_session() -> AsyncSession:
        return session

    app = create_app(
        Settings(
            environment="test",
            api_prefix="/api/v1",
        )
    )
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db_session] = override_db_session

    response = TestClient(app).post(
        f"/api/v1/trips/{trip.id}/generate-plan",
        json={
            "additional_requirements": "优先安排博物馆",
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["summary"] == "澳门文化一日游"
    assert response_data["budget_analysis"] == ("预计总费用为250元，总预算为1000元，剩余750元。")
    assert response_data["total_cost"] == 250
    assert response_data["trip"]["itinerary"][0]["title"] == "澳门历史城区"

    assert trip.itinerary is not None
    assert trip.itinerary[0]["title"] == "澳门历史城区"

    knowledge_search_mock.assert_awaited_once()
    web_search_mock.assert_awaited_once()
    plan_mock.assert_awaited_once()
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(trip)

    plan_arguments = plan_mock.await_args.kwargs

    assert "优先安排博物馆" in plan_arguments["query"]
    assert "澳门旅游指南" in plan_arguments["knowledge"]
    assert "澳门实时资料" in plan_arguments["knowledge"]


def test_create_trip_rejects_reversed_dates() -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)

    response = make_authenticated_client(user, session).post(
        "/api/v1/trips",
        json={
            "title": "日期错误的澳门行程",
            "start_date": "2026-10-03",
            "end_date": "2026-10-01",
        },
    )

    assert response.status_code == 422
    assert "end_date must not be before start_date" in response.text
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


def test_get_trip_hides_trip_not_owned_by_current_user() -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None
    trip_id = uuid4()

    response = make_authenticated_client(user, session).get(f"/api/v1/trips/{trip_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Trip not found"}

    statement = session.scalar.await_args.args[0]
    parameters = statement.compile().params.values()
    assert trip_id in parameters
    assert user.id in parameters
    assert TARGET_CITY in parameters


def test_update_trip_changes_allowed_fields() -> None:
    user = User(id=uuid4())
    trip = make_trip(user.id)
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = trip

    response = make_authenticated_client(user, session).patch(
        f"/api/v1/trips/{trip.id}",
        json={
            "title": "澳门美食文化之旅",
            "budget": 1800,
            "status": "confirmed",
        },
    )

    assert response.status_code == 200
    assert response.json()["title"] == "澳门美食文化之旅"
    assert response.json()["budget"] == 1800
    assert response.json()["status"] == "confirmed"
    assert trip.destination == TARGET_CITY
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(trip)


def test_delete_trip_deletes_owned_trip() -> None:
    user = User(id=uuid4())
    trip = make_trip(user.id)
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = trip

    response = make_authenticated_client(user, session).delete(f"/api/v1/trips/{trip.id}")

    assert response.status_code == 204
    assert response.content == b""
    session.delete.assert_awaited_once_with(trip)
    session.commit.assert_awaited_once()


def test_generate_trip_plan_returns_404_for_unavailable_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None
    get_orchestrator = pytest.fail
    monkeypatch.setattr(
        "app.api.trip_routes.get_travel_orchestrator",
        get_orchestrator,
    )

    response = make_authenticated_client(user, session).post(
        f"/api/v1/trips/{uuid4()}/generate-plan",
        json={},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Trip not found"}


def test_generate_trip_plan_returns_503_when_configuration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    trip = make_trip(user.id)
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = trip

    def raise_configuration_error() -> None:
        raise ValueError("DeepSeek API key is not configured")

    monkeypatch.setattr(
        "app.api.trip_routes.get_travel_orchestrator",
        raise_configuration_error,
    )

    response = make_authenticated_client(user, session).post(
        f"/api/v1/trips/{trip.id}/generate-plan",
        json={},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "DeepSeek API key is not configured",
    }
    session.commit.assert_not_awaited()


def test_generate_trip_plan_returns_502_when_rag_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(id=uuid4())
    trip = make_trip(user.id)
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = trip
    orchestrator = SimpleNamespace(
        search_web=AsyncMock(),
        trip_agent=SimpleNamespace(plan=AsyncMock()),
    )
    monkeypatch.setattr(
        "app.api.trip_routes.get_travel_orchestrator",
        lambda: orchestrator,
    )
    monkeypatch.setattr(
        "app.api.trip_routes.search_reranked_knowledge",
        AsyncMock(side_effect=RuntimeError("vector store failed")),
    )

    response = make_authenticated_client(user, session).post(
        f"/api/v1/trips/{trip.id}/generate-plan",
        json={},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Trip planning service is unavailable",
    }
    session.commit.assert_not_awaited()


def test_apply_trip_dates_fills_dates_and_numbers() -> None:
    trip = Trip(
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 1),
    )
    plan = GeneratedItinerary(
        summary="澳门一日游",
        days=[
            {
                "day_number": 9,
                "travel_date": None,
                "title": "澳门文化之旅",
                "items": [
                    {
                        "time": "09:00",
                        "type": "activity",
                        "description": "参观澳门博物馆",
                        "cost": 0,
                    }
                ],
            }
        ],
    )

    apply_trip_dates(trip, plan)

    assert plan.days[0].day_number == 1
    assert plan.days[0].travel_date == date(
        2026,
        10,
        1,
    )

    trip.end_date = date(2026, 10, 2)

    with pytest.raises(HTTPException) as error:
        apply_trip_dates(trip, plan)

    assert error.value.status_code == 502


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (
            1000,
            "预计总费用为250元，总预算为1000元，剩余750元。",
        ),
        (
            100,
            "预计总费用为250元，总预算为100元，超出预算150元。",
        ),
        (
            None,
            "预计总费用为250元，当前行程未设置总预算。",
        ),
    ],
)
def test_build_budget_analysis(
    budget: int | None,
    expected: str,
) -> None:
    assert build_budget_analysis(250, budget) == expected
