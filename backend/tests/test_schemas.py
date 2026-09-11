from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import (
    ChatRequest,
    ChatResponse,
    GeneratedItinerary,
    TokenResponse,
    TripCreate,
    UserCreate,
    UserLogin,
)


def test_user_create_accepts_valid_data() -> None:
    user = UserCreate(
        email="alice@example.com",
        password="strong-password",
        display_name="  Alice  ",
    )

    assert str(user.email) == "alice@example.com"
    assert user.display_name == "Alice"


def test_user_create_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        UserCreate(
            email="alice@example.com",
            password="short",
        )


def test_user_create_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        UserCreate(
            email="bob@example.com", password="strong-password", is_active=False, role="admin"
        )


def test_user_login_accepts_valid_data() -> None:
    login_data = UserLogin(
        email="alice@example.com",
        password="strong-password",
    )

    assert str(login_data.email) == "alice@example.com"
    assert login_data.password == "strong-password"


def test_token_response_defaults_to_bearer() -> None:
    token = TokenResponse(access_token="test-token")

    assert token.access_token == "test-token"
    assert token.token_type == "bearer"


def test_trip_create_accepts_valid_data() -> None:
    trip = TripCreate(
        title="国庆澳门之旅",
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 4),
        budget=5000,
        num_people=2,
        preferences={"interests": ["美食", "博物馆"]},
    )

    assert trip.title == "国庆澳门之旅"
    assert trip.num_people == 2
    assert trip.preferences["interests"] == ["美食", "博物馆"]


def test_trip_create_rejects_invalid_numbers() -> None:
    with pytest.raises(ValidationError):
        TripCreate(
            title="无效旅行",
            budget=-1,
            num_people=0,
        )


def test_trip_create_rejects_custom_destination() -> None:
    with pytest.raises(ValidationError):
        TripCreate(
            title="其他城市旅行",
            destination="上海",
        )


def test_chat_request_accepts_valid_message() -> None:
    chat_data = ChatRequest(message="  我想去澳门旅行  ")

    assert chat_data.message == "我想去澳门旅行"


def test_chat_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_accepts_conversation_id() -> None:
    conversation_id = uuid4()

    chat_data = ChatRequest(
        message="继续规划第二天",
        conversation_id=conversation_id,
    )

    assert chat_data.conversation_id == conversation_id


def test_chat_response_contains_conversation_id() -> None:
    conversation_id = uuid4()

    response = ChatResponse(
        reply="这是第二天的安排",
        conversation_id=conversation_id,
    )

    assert response.reply == "这是第二天的安排"
    assert response.conversation_id == conversation_id


def test_generated_itinerary_accepts_valid_data() -> None:
    itinerary = GeneratedItinerary(
        summary="澳门三日文化美食之旅",
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
                        "cost": 0,
                    },
                    {
                        "time": "12:00",
                        "type": "food",
                        "description": "品尝澳门本地美食",
                        "cost": 150,
                    },
                ],
            }
        ],
    )

    assert itinerary.days[0].day_number == 1
    assert itinerary.days[0].travel_date == date(2026, 10, 1)
    assert itinerary.days[0].items[1].cost == 150


def test_generated_itinerary_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        GeneratedItinerary(
            summary="无效行程",
            days=[
                {
                    "day_number": 1,
                    "title": "第一天",
                    "items": [
                        {
                            "time": "09:00",
                            "type": "activity",
                            "description": "参观景点",
                            "cost": -100,
                        }
                    ],
                }
            ],
        )


def test_trip_create_rejects_reversed_date_range() -> None:
    with pytest.raises(
        ValidationError,
        match="end_date must not be before start_date",
    ):
        TripCreate(
            title="日期错误的澳门行程",
            start_date=date(2026, 10, 3),
            end_date=date(2026, 10, 1),
        )
