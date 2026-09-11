import json
import logging
from datetime import date
from functools import lru_cache
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tavily import AsyncTavilyClient

from app.agents.trip_agent import TripAgent
from app.config import TARGET_CITY, get_settings
from app.models import Trip
from app.rag.rag_service import search_reranked_knowledge

logger = logging.getLogger("uvicorn.error")

SYSTEM_PROMPT = (
    "你是专门服务澳门特别行政区的旅行规划助手，请优先使用中文回答。"
    "你的服务范围仅限澳门；如果用户要求规划澳门以外的目的地，"
    "应简短说明当前只支持澳门，不要调用工具。"
    "只能根据工具返回的真实数据回答，不得编造。"
    "请根据问题选择工具："
    "用户明确询问自己已保存的行程时，调用 list_my_trips；"
    "普通澳门景点推荐不需要查询用户行程。"
    "查询历史、文化、景点介绍和静态攻略时，"
    "必须调用 search_travel_knowledge。"
    "调用 search_travel_knowledge 后，最终回答末尾必须列出“资料来源”。"
    "每条来源必须使用 Markdown 格式："
    "`[资料标题，第X页](source_url)`。"
    "资料标题、页码和URL必须直接复制工具结果中的"
    "title、page_number和source_url，不能自行编写，"
    "不得只写资料名称或发布机构，也不得省略URL。"
    "查询今天、明天、最新、开放时间、门票、预约、天气等"
    "实时信息时，必须调用 search_web。"
    "调用 search_web 后，最终回答末尾必须列出“网页来源”，"
    "每条来源使用 Markdown 格式：`[网页标题](url)`，"
    "标题和URL必须直接复制工具结果。"
    "如果网页资料日期已经过期，不能作为当前信息推荐。"
    "如果用户明确要求只根据知识库回答，只能调用"
    "search_travel_knowledge。"
    "制定完整行程时，先查询相关资料，再调用 plan_trip。"
    "澳门的酒店、美食和交通资料也应先通过知识库或网页搜索取得，"
    "再由 plan_trip 统一整理进完整行程。"
    "plan_trip 会对初稿进行自我反思，"
    "在评分不足或规则检查失败时最多修订三次；"
    "回答时需要说明是否发生了修订。"
)


class TravelOrchestrator:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key is not configured")

        if not settings.tavily_api_key:
            raise ValueError("Tavily API key is not configured")

        self.model = ChatDeepSeek(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            extra_body={"thinking": {"type": "disabled"}},
        )

        self.web_search_client = AsyncTavilyClient(api_key=settings.tavily_api_key)

        self.trip_agent = TripAgent()
        self.checkpoint_database_url = settings.database_url.replace(
            "postgresql+asyncpg://",
            "postgresql://",
            1,
        )

    async def search_web(self, query: str) -> list[dict[str, str]]:
        scoped_query = query if TARGET_CITY in query else f"{TARGET_CITY} {query}"
        response = await self.web_search_client.search(
            query=scoped_query,
            search_depth="advanced",
            max_results=5,
            include_answer=False,
        )

        return [
            {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
            }
            for result in response["results"]
        ]

    async def answer(
        self,
        message: str,
        user_id: UUID,
        conversation_id: UUID,
        session: AsyncSession,
    ) -> str:
        @tool
        async def list_my_trips() -> str:
            """查询当前登录用户最近的行程，包括日期、预算、状态和偏好。"""
            logger.info("Travel agent calling tool: list_my_trips")

            result = await session.scalars(
                select(Trip)
                .where(
                    Trip.user_id == user_id,
                    Trip.destination == TARGET_CITY,
                )
                .order_by(Trip.created_at.desc())
                .limit(10)
            )
            trips = list(result)

            return json.dumps(
                {
                    "count": len(trips),
                    "trips": [
                        {
                            "id": str(trip.id),
                            "title": trip.title,
                            "destination": trip.destination,
                            "country": trip.country,
                            "start_date": trip.start_date,
                            "end_date": trip.end_date,
                            "budget": trip.budget,
                            "num_people": trip.num_people,
                            "status": trip.status,
                            "preferences": trip.preferences,
                            "itinerary": trip.itinerary,
                        }
                        for trip in trips
                    ],
                },
                ensure_ascii=False,
                default=str,
            )

        @tool
        async def search_travel_knowledge(query: str) -> str:
            """按照自然语言问题检索澳门旅行知识库。"""
            logger.info(
                "Travel agent calling tool: search_travel_knowledge, query=%s",
                query,
            )

            scoped_query = query if TARGET_CITY in query else f"{TARGET_CITY} {query}"
            results = await search_reranked_knowledge(scoped_query)

            return json.dumps(
                {
                    "count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            )

        @tool
        async def search_web(query: str) -> str:
            """搜索澳门开放时间、票价、通关、天气等实时旅行信息。"""

            logger.info(
                "Travel agent calling tool: search_web, query=%s",
                query,
            )

            results = await self.search_web(query)

            return json.dumps(
                {
                    "count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            )

        @tool
        async def plan_trip(
            query: str,
            knowledge: str = "",
        ) -> str:
            """使用已经查询到的真实资料生成、自评并修订逐日行程。"""
            logger.info("Travel orchestrator calling TripAgent")

            result = await self.trip_agent.plan(query, knowledge)

            total_cost = sum(item.cost for day in result.plan.days for item in day.items)

            return json.dumps(
                {
                    "plan": result.plan.model_dump(mode="json"),
                    "reflection": result.reflection.model_dump(),
                    "revised": result.revised,
                    "total_cost": total_cost,
                },
                ensure_ascii=False,
            )

        tools: list[BaseTool] = [
            list_my_trips,
            search_travel_knowledge,
            search_web,
            plan_trip,
        ]

        model_with_tools = self.model.bind_tools(tools)

        async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            response = await model_with_tools.ainvoke(
                [
                    SystemMessage(f"当前日期是 {date.today().isoformat()}。{SYSTEM_PROMPT}"),
                    *state["messages"],
                ]
            )
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        thread_id = f"{user_id}:{conversation_id}"

        async with AsyncPostgresSaver.from_conn_string(
            self.checkpoint_database_url
        ) as checkpointer:
            graph = workflow.compile(
                checkpointer=checkpointer,
            )

            result = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=message),
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    },
                    "recursion_limit": 8,
                },
            )

        final_message = result["messages"][-1]

        if not isinstance(final_message, AIMessage):
            raise RuntimeError("Travel agent did not return an AI message")
        if not isinstance(final_message.content, str):
            raise RuntimeError("Unexpected DeepSeek response format")
        return final_message.content


@lru_cache
def get_travel_orchestrator() -> TravelOrchestrator:
    return TravelOrchestrator()
