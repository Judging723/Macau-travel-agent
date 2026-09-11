import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.trip_agent import TripAgent
from app.schemas import GeneratedItinerary, PlanReflection


def make_plan(
    summary: str,
    activity_count: int = 1,
) -> GeneratedItinerary:
    return GeneratedItinerary(
        summary=summary,
        days=[
            {
                "day_number": 1,
                "title": "澳门文化之旅",
                "items": [
                    {
                        "time": "09:00",
                        "type": "activity",
                        "description": f"参观景点{index + 1}",
                        "cost": 0,
                    }
                    for index in range(activity_count)
                ],
            }
        ],
    )


def test_good_draft_is_returned_without_revision() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        draft = make_plan("合格初稿")
        reflection = PlanReflection(
            score=8,
            feedback="计划合理",
            issues=[],
        )

        agent._write_plan = AsyncMock(return_value=draft)
        agent._reflect = AsyncMock(return_value=reflection)
        agent._revise = AsyncMock()

        result = await agent.plan("用户需求", "真实资料")

        assert result.plan is draft
        assert result.reflection is reflection
        assert result.revised is False
        agent._revise.assert_not_awaited()

    asyncio.run(run())


def test_low_score_draft_is_revised_once() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        draft = make_plan("需要修改的初稿")
        revised_plan = make_plan("修改后的计划")
        reflection = PlanReflection(
            score=6,
            feedback="行程太紧凑",
            issues=["减少第一天的活动数量"],
        )

        agent._write_plan = AsyncMock(return_value=draft)
        agent._reflect = AsyncMock(return_value=reflection)
        agent._revise = AsyncMock(return_value=revised_plan)

        result = await agent.plan("用户需求", "真实资料")

        assert result.plan is revised_plan
        assert result.reflection is reflection
        assert result.revised is True
        agent._revise.assert_awaited_once_with(
            "用户需求",
            "真实资料",
            draft,
            reflection,
        )

    asyncio.run(run())


def test_activity_limit_forces_revision() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        draft = make_plan("超限初稿", activity_count=5)
        revised_plan = make_plan("修订行程", activity_count=4)
        reflection = PlanReflection(
            score=9,
            feedback="模型认为计划合理",
            issues=[],
        )

        agent._write_plan = AsyncMock(return_value=draft)
        agent._reflect = AsyncMock(return_value=reflection)
        agent._revise = AsyncMock(return_value=revised_plan)

        result = await agent.plan("普通澳门行程", "真实资料")

        assert result.plan is revised_plan
        assert result.revised is True

        revised_reflection = agent._revise.await_args.args[3]
        assert revised_reflection.score == 7
        assert "第1天有5个activity" in revised_reflection.issues[-1]

    asyncio.run(run())


def test_relaxed_trip_rejects_four_activities_after_max_revisions() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        overloaded_plan = make_plan("长辈行程", activity_count=4)
        reflection = PlanReflection(
            score=9,
            feedback="模型认为计划合理",
            issues=[],
        )

        agent._write_plan = AsyncMock(return_value=overloaded_plan)
        agent._reflect = AsyncMock(return_value=reflection)
        agent._revise = AsyncMock(return_value=overloaded_plan)

        with pytest.raises(
            RuntimeError,
            match="exceeds activity limit",
        ):
            await agent.plan("长辈同行，不要太满", "真实资料")

        assert agent._revise.await_count == 3
        revised_reflection = agent._revise.await_args_list[0].args[3]
        assert "最多只能有3个" in revised_reflection.issues[-1]

    asyncio.run(run())


def test_relaxed_trip_can_pass_on_second_revision() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        overloaded_plan = make_plan("长辈行程", activity_count=4)
        valid_plan = make_plan("修订后的长辈行程", activity_count=3)
        reflection = PlanReflection(
            score=9,
            feedback="模型认为计划合理",
            issues=[],
        )

        agent._write_plan = AsyncMock(return_value=overloaded_plan)
        agent._reflect = AsyncMock(return_value=reflection)
        agent._revise = AsyncMock(side_effect=[overloaded_plan, valid_plan])

        result = await agent.plan("长辈同行，不要太满", "真实资料")

        assert result.plan is valid_plan
        assert result.revised is True
        assert agent._revise.await_count == 2

    asyncio.run(run())


def test_reflect_builds_messages_and_returns_reflection() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        agent.reflection_prompt = "测试反思提示词"
        agent.reflection_model = AsyncMock()

        expected_reflection = PlanReflection(
            score=7,
            feedback="需要减少活动数量",
            issues=["第一天安排过多"],
        )
        agent.reflection_model.ainvoke.return_value = expected_reflection

        draft = make_plan("待反思的初稿")

        result = await agent._reflect(
            query="预算3000元，两人同行",
            knowledge="澳门旅游资料",
            draft=draft,
        )

        assert result is expected_reflection
        agent.reflection_model.ainvoke.assert_awaited_once()

        messages = agent.reflection_model.ainvoke.await_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == "测试反思提示词"
        assert "预算3000元，两人同行" in messages[1].content
        assert "澳门旅游资料" in messages[1].content
        assert "待反思的初稿" in messages[1].content

    asyncio.run(run())


def test_revise_builds_messages_and_returns_revised_plan() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        agent.plan_prompt = "测试行程规划提示词"
        agent.plan_model = AsyncMock()

        draft = make_plan("原始计划")
        reflection = PlanReflection(
            score=6,
            feedback="行程安排太紧凑",
            issues=["减少第一天的活动数量"],
        )
        expected_plan = make_plan("修改后的计划")

        agent.plan_model.ainvoke.return_value = expected_plan

        result = await agent._revise(
            query="预算3000元，两人同行",
            knowledge="澳门旅游资料",
            draft=draft,
            reflection=reflection,
        )

        assert result is expected_plan
        agent.plan_model.ainvoke.assert_awaited_once()

        messages = agent.plan_model.ainvoke.await_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == "测试行程规划提示词"
        assert "预算3000元，两人同行" in messages[1].content
        assert "澳门旅游资料" in messages[1].content
        assert "原始计划" in messages[1].content
        assert "行程安排太紧凑" in messages[1].content
        assert "减少第一天的活动数量" in messages[1].content
        assert "每天type=activity的item不得超过4个" in messages[1].content

    asyncio.run(run())


def test_revise_rejects_invalid_model_output() -> None:
    async def run() -> None:
        agent = TripAgent.__new__(TripAgent)
        agent.plan_prompt = "测试行程规划提示词"
        agent.plan_model = AsyncMock()
        agent.plan_model.ainvoke.return_value = "无效结果"

        draft = make_plan("原始计划")
        reflection = PlanReflection(
            score=5,
            feedback="需要修改",
            issues=["行程太紧凑"],
        )

        with pytest.raises(
            RuntimeError,
            match="DeepSeek did not return a valid revised itinerary",
        ):
            await agent._revise(
                query="预算3000元",
                knowledge="澳门旅游 merr资料",
                draft=draft,
                reflection=reflection,
            )

    asyncio.run(run())
