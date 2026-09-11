import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.config import get_settings
from app.schemas import GeneratedItinerary, PlanReflection

logger = logging.getLogger(__name__)

RELAXED_TRIP_TERMS = (
    "长辈",
    "舒缓",
    "轻松",
    "不要太满",
)
MAX_PLAN_REVISIONS = 3


@dataclass(frozen=True)
class ReflectedPlan:
    plan: GeneratedItinerary
    reflection: PlanReflection
    revised: bool


def get_activity_limit(query: str) -> int:
    if any(term in query for term in RELAXED_TRIP_TERMS):
        return 3

    return 4


def find_activity_limit_issues(
    plan: GeneratedItinerary,
    limit: int,
) -> list[str]:
    issues = []

    for day in plan.days:
        activity_count = sum(item.type == "activity" for item in day.items)

        if activity_count > limit:
            issues.append(
                f"第{day.day_number}天有{activity_count}个activity，"
                f"最多只能有{limit}个；"
                "请合并或删减游览项目，"
                "休息不能标记为activity，"
                "交通移动必须标记为transport。"
            )

    return issues


def build_activity_limit_instruction(query: str) -> str:
    limit = get_activity_limit(query)
    return (
        f"硬性约束：每天type=activity的item不得超过{limit}个。"
        "请在输出前逐日计数。"
        "纯休息不要创建item，交通移动必须使用transport。"
    )


class TripAgent:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key is not configured")

        self.model = ChatDeepSeek(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            extra_body={"thinking": {"type": "disabled"}},
        )

        self.plan_model = self.model.with_structured_output(
            GeneratedItinerary,
            method="function_calling",
        )
        self.reflection_model = self.model.with_structured_output(
            PlanReflection,
            method="function_calling",
        )

        self.plan_prompt = (
            "你是一名澳门特别行政区旅行行程规划专家。"
            "只能规划澳门范围内的地点和活动，不得加入其他目的地。"
            "请根据用户需求和提供的真实旅行资料生成逐日行程。"
            "每天需要包含上午、午餐、下午、晚间或交通安排。"
            "item 的 type 只能是 activity、food、transport、hotel。"
            "cost 必须使用非负整数，单位为人民币元。"
            "安排需要考虑时间、地点距离、用户预算和同行人数。"
            "普通行程每天的type=activity核心游览项目不得超过4项；"
            "如果用户提到长辈同行、舒缓、轻松或不要太满，"
            "每天的type=activity不得超过3项。"
            "用餐和交通需要单独记录；"
            "交通移动必须标记为transport。"
            "可以在时间安排中预留休息，"
            "但不要把纯休息另建为type=activity的item。"
            "资料不足时需要明确说明，禁止编造不存在的地点、酒店、餐厅、班次和价格。"
            "每个item的cost表示当前整个同行团队的该项总费用，"
            "不是单人费用。"
            "summary只描述行程主题和路线，不得填写总费用、"
            "剩余预算或预算结论；这些数据由后端程序计算。"
        )
        self.reflection_prompt = (
            "请反思你刚才生成的旅行计划，并按照1至10分评分。"
            "检查计划是否符合用户预算、天数和偏好，日程是否连贯，"
            "所有地点是否位于澳门范围内，地点距离和费用是否合理，"
            "是否使用了资料之外的信息。"
            "feedback 给出简短总结，issues 列出需要修改的问题。"
            "逐日统计type=activity的数量；"
            "普通行程超过4项，或长辈舒缓行程超过3项时，"
            "必须列入issues，并将评分设为8分以下。"
        )

    async def plan(self, query: str, knowledge: str = "") -> ReflectedPlan:
        draft = await self._write_plan(query, knowledge)
        reflection = await self._reflect(query, knowledge, draft)

        activity_limit = get_activity_limit(query)
        rule_issues = find_activity_limit_issues(
            draft,
            activity_limit,
        )

        if rule_issues:
            reflection = reflection.model_copy(
                update={
                    "score": min(reflection.score, 7),
                    "issues": [
                        *reflection.issues,
                        *rule_issues,
                    ],
                }
            )

        if reflection.score >= 8:
            return ReflectedPlan(
                plan=draft,
                reflection=reflection,
                revised=False,
            )

        current_plan = draft

        for _ in range(MAX_PLAN_REVISIONS):
            current_plan = await self._revise(
                query,
                knowledge,
                current_plan,
                reflection,
            )
            remaining_issues = find_activity_limit_issues(
                current_plan,
                activity_limit,
            )

            if not remaining_issues:
                return ReflectedPlan(
                    plan=current_plan,
                    reflection=reflection,
                    revised=True,
                )

            reflection = reflection.model_copy(
                update={
                    "score": 7,
                    "issues": remaining_issues,
                }
            )

        raise RuntimeError("DeepSeek revised itinerary exceeds activity limit")

    async def _write_plan(
        self,
        query: str,
        knowledge: str,
    ) -> GeneratedItinerary:
        text = f"用户需求：{query}"

        if knowledge:
            text += f"\n\n旅行资料：\n{knowledge}"

        text += f"\n\n{build_activity_limit_instruction(query)}"

        messages = [
            SystemMessage(self.plan_prompt),
            HumanMessage(text),
        ]

        result = await self.plan_model.ainvoke(messages)

        if not isinstance(result, GeneratedItinerary):
            raise RuntimeError("DeepSeek did not return a valid itinerary")

        return result

    async def _reflect(
        self,
        query: str,
        knowledge: str,
        draft: GeneratedItinerary,
    ) -> PlanReflection:
        text = (
            f"用户需求：{query}\n\n"
            f"旅行资料：\n{knowledge or '没有额外资料'}\n\n"
            f"你生成的计划：\n{draft.model_dump_json(indent=2)}"
        )

        result = await self.reflection_model.ainvoke(
            [
                SystemMessage(self.reflection_prompt),
                HumanMessage(text),
            ]
        )

        if not isinstance(result, PlanReflection):
            raise RuntimeError("DeepSeek did not return a valid reflection")

        return result

    async def _revise(
        self,
        query: str,
        knowledge: str,
        draft: GeneratedItinerary,
        reflection: PlanReflection,
    ) -> GeneratedItinerary:
        text = (
            f"用户需求：{query}\n\n"
            f"旅行资料：\n{knowledge or '没有额外资料'}\n\n"
            f"原计划：\n{draft.model_dump_json(indent=2)}\n\n"
            f"自我反思：\n{reflection.model_dump_json(indent=2)}\n\n"
            "请根据自我反思修正原计划，只使用已有资料，"
            "不要增加未经证实的信息。\n\n"
            f"{build_activity_limit_instruction(query)}"
        )

        result = await self.plan_model.ainvoke(
            [
                SystemMessage(self.plan_prompt),
                HumanMessage(text),
            ]
        )

        if not isinstance(result, GeneratedItinerary):
            logger.error(
                "Invalid revised itinerary: type=%s, value=%r",
                type(result).__name__,
                result,
            )
            raise RuntimeError("DeepSeek did not return a valid revised itinerary")

        return result
