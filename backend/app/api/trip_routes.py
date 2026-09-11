import json
import logging
from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.agents.orchestrator import get_travel_orchestrator
from app.api.auth_routes import CurrentUser, DatabaseSession
from app.config import TARGET_CITY, TARGET_COUNTRY
from app.models import Trip
from app.rag.rag_service import search_reranked_knowledge
from app.schemas import (
    GeneratedItinerary,
    TripCreate,
    TripPlanRequest,
    TripPlanResponse,
    TripResponse,
    TripUpdate,
)

router = APIRouter(
    prefix="/trips",
    tags=["trips"],
)

logger = logging.getLogger(__name__)


def apply_trip_dates(
    trip: Trip,
    plan: GeneratedItinerary,
) -> None:
    if trip.start_date is None or trip.end_date is None:
        return

    expected_day_count = (trip.end_date - trip.start_date).days + 1

    if len(plan.days) != expected_day_count:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("Generated itinerary day count does not match trip dates"),
        )

    for index, day in enumerate(plan.days):
        day.day_number = index + 1
        day.travel_date = trip.start_date + timedelta(days=index)


def build_budget_analysis(
    total_cost: int,
    budget: int | None,
) -> str:
    if budget is None:
        return f"预计总费用为{total_cost}元，当前行程未设置总预算。"

    difference = budget - total_cost

    if difference >= 0:
        return f"预计总费用为{total_cost}元，总预算为{budget}元，剩余{difference}元。"

    return f"预计总费用为{total_cost}元，总预算为{budget}元，超出预算{-difference}元。"


@router.post(
    "",
    response_model=TripResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_trip(
    trip_data: TripCreate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Trip:
    trip = Trip(
        user_id=current_user.id,
        destination=TARGET_CITY,
        country=TARGET_COUNTRY,
        **trip_data.model_dump(),
    )

    session.add(trip)
    await session.commit()
    await session.refresh(trip)

    return trip


@router.post(
    "/{trip_id}/generate-plan",
    response_model=TripPlanResponse,
)
async def generate_trip_plan(
    trip_id: UUID,
    plan_data: TripPlanRequest,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> TripPlanResponse:
    trip = await session.scalar(
        select(Trip).where(
            Trip.id == trip_id,
            Trip.user_id == current_user.id,
            Trip.destination == TARGET_CITY,
        )
    )

    if trip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trip not found",
        )

    preferences = json.dumps(
        trip.preferences or {},
        ensure_ascii=False,
    )

    query_parts = [
        f"行程名称：{trip.title}",
        f"目的地：{TARGET_CITY}",
        f"国家或地区：{TARGET_COUNTRY}",
        f"开始日期：{trip.start_date or '未确定'}",
        f"结束日期：{trip.end_date or '未确定'}",
        f"总预算：{trip.budget if trip.budget is not None else '未设置'}",
        f"同行人数：{trip.num_people}",
        f"旅行偏好：{preferences}",
    ]

    if plan_data.additional_requirements:
        query_parts.append(f"补充要求：{plan_data.additional_requirements}")

    query = "\n".join(query_parts)

    try:
        orchestrator = get_travel_orchestrator()

        knowledge_results = await search_reranked_knowledge(
            f"{TARGET_CITY} 景点 美食 住宿 交通 旅行攻略"
        )
        web_results = await orchestrator.search_web(
            f"{TARGET_CITY} 最新 景点 美食 酒店 交通 旅行信息"
        )

        knowledge = json.dumps(
            {
                "knowledge_base": knowledge_results,
                "web_search": web_results,
            },
            ensure_ascii=False,
        )

        result = await orchestrator.trip_agent.plan(
            query=query,
            knowledge=knowledge,
        )
        plan = result.plan

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Trip plan generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Trip planning service is unavailable",
        ) from exc

    apply_trip_dates(trip, plan)

    trip.itinerary = [day.model_dump(mode="json") for day in plan.days]

    await session.commit()
    await session.refresh(trip)

    total_cost = sum(item.cost for day in plan.days for item in day.items)

    return TripPlanResponse(
        trip=TripResponse.model_validate(trip),
        summary=plan.summary,
        budget_analysis=build_budget_analysis(
            total_cost,
            trip.budget,
        ),
        total_cost=total_cost,
    )


@router.get(
    "",
    response_model=list[TripResponse],
)
async def list_trips(
    current_user: CurrentUser,
    session: DatabaseSession,
) -> list[Trip]:
    result = await session.scalars(
        select(Trip)
        .where(
            Trip.user_id == current_user.id,
            Trip.destination == TARGET_CITY,
        )
        .order_by(Trip.created_at.desc())
    )

    return list(result)


@router.get(
    "/{trip_id}",
    response_model=TripResponse,
)
async def get_trip(
    trip_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Trip:
    trip = await session.scalar(
        select(Trip).where(
            Trip.id == trip_id,
            Trip.user_id == current_user.id,
            Trip.destination == TARGET_CITY,
        )
    )

    if trip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trip not found",
        )

    return trip


@router.patch(
    "/{trip_id}",
    response_model=TripResponse,
)
async def update_trip(
    trip_id: UUID,
    trip_data: TripUpdate,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> Trip:
    trip = await session.scalar(
        select(Trip).where(
            Trip.id == trip_id,
            Trip.user_id == current_user.id,
            Trip.destination == TARGET_CITY,
        )
    )

    if trip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trip not found",
        )

    updates = trip_data.model_dump(
        exclude_unset=True,
        exclude_none=True,
    )

    for field, value in updates.items():
        setattr(trip, field, value)

    await session.commit()
    await session.refresh(trip)

    return trip


@router.delete(
    "/{trip_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_trip(
    trip_id: UUID,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> None:
    trip = await session.scalar(
        select(Trip).where(
            Trip.id == trip_id,
            Trip.user_id == current_user.id,
            Trip.destination == TARGET_CITY,
        )
    )

    if trip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trip not found",
        )

    await session.delete(trip)
    await session.commit()
