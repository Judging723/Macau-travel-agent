import json

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.orchestrator import apply_verified_sources


def make_tool_message(name: str, results: list[dict]) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        ),
        tool_call_id=f"{name}-call",
        name=name,
    )


def test_apply_verified_sources_rebuilds_knowledge_citations() -> None:
    message = make_tool_message(
        "search_travel_knowledge",
        [
            {
                "title": "澳门历史简介",
                "page_number": 1,
                "source_url": "https://example.com/history.pdf",
            }
        ],
    )
    answer = (
        "MACAU 与妈阁有关。\n\n"
        "**资料来源**\n"
        "- [虚构来源](https://invalid.example/source)"
    )

    result = apply_verified_sources(answer, [message])

    assert "https://invalid.example/source" not in result
    assert "[澳门历史简介，第1页](https://example.com/history.pdf)" in result
    assert result.count("资料来源") == 1


def test_apply_verified_sources_refuses_when_knowledge_has_no_evidence() -> None:
    message = make_tool_message("search_travel_knowledge", [])

    result = apply_verified_sources("模型自行生成的答案", [message])

    assert "没有足够资料" in result
    assert "模型自行生成的答案" not in result


def test_apply_verified_sources_keeps_answer_without_tool_calls() -> None:
    answer = "当前项目仅支持澳门旅游规划。"

    result = apply_verified_sources(answer, [AIMessage(content=answer)])

    assert result == answer
