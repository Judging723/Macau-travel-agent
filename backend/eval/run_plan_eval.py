import asyncio
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from eval.run_agent_eval import normalize_text

CASES_PATH = Path(__file__).with_name("plan_cases.json")
BASE_URL = os.getenv(
    "TRAVEL_AGENT_EVAL_BASE_URL",
    "http://127.0.0.1:8000",
)

CHECK_LABELS = {
    "day_count": "行程天数",
    "day_numbers": "日期编号",
    "travel_dates": "旅行日期",
    "budget": "预算约束",
    "daily_load": "每日活动数量",
    "item_types": "活动类型",
    "concepts": "偏好概念",
    "scope": "澳门范围",
    "summary_cost": "摘要不含金额",
    "budget_text": "预算文本一致",
}


def get_expected_dates(
    start_date: str,
    end_date: str,
) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    day_count = (end - start).days + 1

    return [(start + timedelta(days=offset)).isoformat() for offset in range(day_count)]


def build_expected_budget_analysis(
    total_cost: int,
    budget: int | None,
) -> str:
    if budget is None:
        return f"预计总费用为{total_cost}元，当前行程未设置总预算。"

    difference = budget - total_cost

    if difference >= 0:
        return f"预计总费用为{total_cost}元，总预算为{budget}元，剩余{difference}元。"

    return f"预计总费用为{total_cost}元，总预算为{budget}元，超出预算{-difference}元。"


def evaluate_plan(
    response_data: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, bool]:
    trip_data = case["trip"]
    expected = case["expected"]
    itinerary = response_data["trip"]["itinerary"] or []

    expected_dates = get_expected_dates(
        trip_data["start_date"],
        trip_data["end_date"],
    )
    actual_dates = [day["travel_date"] for day in itinerary]
    actual_day_numbers = [day["day_number"] for day in itinerary]

    total_cost = response_data["total_cost"]
    budget = trip_data.get("budget")

    item_types = {item["type"] for day in itinerary for item in day["items"]}

    activity_counts = [
        sum(item["type"] == "activity" for item in day["items"]) for day in itinerary
    ]

    plan_text = json.dumps(
        {
            "summary": response_data["summary"],
            "budget_analysis": response_data["budget_analysis"],
            "itinerary": itinerary,
        },
        ensure_ascii=False,
    )
    normalized_plan = normalize_text(plan_text)

    concepts_ok = all(
        any(normalize_text(term) in normalized_plan for term in alternatives)
        for alternatives in expected["required_concepts"]
    )

    scope_ok = not any(
        normalize_text(term) in normalized_plan for term in expected["forbidden_terms"]
    )

    summary_has_no_cost = (
        re.search(
            r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)"
            r"\s*(?:人民币|澳门元|元|mop)",
            response_data["summary"],
            flags=re.IGNORECASE,
        )
        is None
    )

    budget_text_is_consistent = response_data["budget_analysis"] == build_expected_budget_analysis(
        total_cost,
        budget,
    )

    return {
        "day_count": (len(itinerary) == len(expected_dates)),
        "day_numbers": (actual_day_numbers == list(range(1, len(itinerary) + 1))),
        "travel_dates": (actual_dates == expected_dates),
        "budget": (budget is None or total_cost <= budget),
        "daily_load": all(count <= expected["max_activities_per_day"] for count in activity_counts),
        "item_types": set(expected["required_item_types"]).issubset(item_types),
        "concepts": concepts_ok,
        "scope": scope_ok,
        "summary_cost": summary_has_no_cost,
        "budget_text": budget_text_is_consistent,
    }


async def delete_test_trip(
    client: httpx.AsyncClient,
    trip_id: str,
) -> None:
    try:
        response = await client.delete(f"/api/v1/trips/{trip_id}")

        if response.status_code != 204:
            print(f"  [WARN] 清理行程失败：HTTP {response.status_code}")
    except httpx.HTTPError as exc:
        print(f"  [WARN] 清理行程失败：{exc}")


async def main() -> None:
    token = os.getenv("TRAVEL_AGENT_EVAL_TOKEN")

    if not token:
        raise RuntimeError("TRAVEL_AGENT_EVAL_TOKEN is not configured")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    case_id = os.getenv("TRAVEL_AGENT_EVAL_CASE_ID")

    if case_id:
        cases = [case for case in cases if case["id"] == case_id]

        if not cases:
            raise ValueError(f"Unknown evaluation case: {case_id}")

    completed_count = 0
    passed_count = 0
    total_latency_ms = 0.0
    check_passed = {name: 0 for name in CHECK_LABELS}

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=300,
    ) as client:
        for case in cases:
            trip_id: str | None = None
            start = perf_counter()

            try:
                create_response = await client.post(
                    "/api/v1/trips",
                    json=case["trip"],
                )
                create_response.raise_for_status()
                trip_id = create_response.json()["id"]

                plan_response = await client.post(
                    f"/api/v1/trips/{trip_id}/generate-plan",
                    json={
                        "additional_requirements": case["additional_requirements"],
                    },
                )
                plan_response.raise_for_status()
                response_data = plan_response.json()

                latency_ms = (perf_counter() - start) * 1000
                total_latency_ms += latency_ms
                completed_count += 1

                checks = evaluate_plan(
                    response_data,
                    case,
                )
                passed = all(checks.values())
                passed_count += int(passed)

                for name, result in checks.items():
                    check_passed[name] += int(result)

                status = "PASS" if passed else "FAIL"

                print(
                    f"[{status}] {case['id']} | "
                    f"cost={response_data['total_cost']} | "
                    f"latency={latency_ms:.0f}ms"
                )

                failed_checks = [
                    CHECK_LABELS[name] for name, result in checks.items() if not result
                ]

                if failed_checks:
                    print("  未通过：", failed_checks)
                    print(
                        json.dumps(
                            response_data,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

            except (
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                latency_ms = (perf_counter() - start) * 1000
                total_latency_ms += latency_ms

                print(f"[ERROR] {case['id']} | latency={latency_ms:.0f}ms | {exc}")

            finally:
                if trip_id is not None:
                    await delete_test_trip(
                        client,
                        trip_id,
                    )

    case_count = len(cases)

    print()
    print(f"用例数量：{case_count}")
    print(f"成功生成：{completed_count}")

    if completed_count:
        print(f"行程通过率：{passed_count / completed_count:.2%}")

        for name, label in CHECK_LABELS.items():
            print(f"{label}通过率：{check_passed[name] / completed_count:.2%}")

    print(f"端到端平均延迟：{total_latency_ms / case_count:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
