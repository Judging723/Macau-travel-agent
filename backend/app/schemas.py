from datetime import date, datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    display_name: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserLogin(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class TripCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    title: str = Field(min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    budget: int | None = Field(default=None, ge=0)
    num_people: int = Field(default=1, ge=1, le=100)
    preferences: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("end_date must not be before start_date")

        return self


class ItineraryItem(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    time: str = Field(min_length=1, max_length=50)
    type: Literal["activity", "food", "transport", "hotel"]
    description: str = Field(min_length=1, max_length=500)
    cost: int = Field(ge=0)


class ItineraryDay(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    day_number: int = Field(ge=1)
    travel_date: date | None = None
    title: str = Field(min_length=1, max_length=200)
    items: list[ItineraryItem] = Field(min_length=1)


class GeneratedItinerary(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    summary: str = Field(min_length=1, max_length=1000)
    days: list[ItineraryDay] = Field(min_length=1)


class PlanReflection(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    score: int = Field(ge=1, le=10)
    feedback: str = Field(min_length=1, max_length=1000)
    issues: list[str] = Field(default_factory=list)


class TripPlanRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    additional_requirements: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
    )


class TripUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    itinerary: list[ItineraryDay] | None = None
    status: Literal["draft", "confirmed", "completed"] | None = None
    budget: int | None = Field(default=None, ge=0)


class TripResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    destination: str
    country: str | None
    start_date: date | None
    end_date: date | None
    budget: int | None
    num_people: int
    status: Literal["draft", "confirmed", "completed"]
    itinerary: list[ItineraryDay] | None
    preferences: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class TripPlanResponse(BaseModel):
    trip: TripResponse
    summary: str
    budget_analysis: str
    total_cost: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    conversation_id: UUID
