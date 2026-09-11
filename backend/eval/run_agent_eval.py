import asyncio
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from opencc import OpenCC

CASES_PATH = Path(__file__).with_name("agent_cases.json")
BASE_URL = os.getenv(
    "TRAVEL_AGENT_EVAL_BASE_URL",
    "http://127.0.0.1:8000",
)
TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def normalize_text(text: str) -> str:
    simplified = TRADITIONAL_TO_SIMPLIFIED.convert(text)
    return "".join(simplified.lower().split())


def evaluate_answer(
    answer: str,
    case: dict[str, Any],
) -> dict[str, bool]:
    normalized_answer = normalize_text(answer)

    concepts_ok = all(
        any(normalize_text(term) in normalized_answer for term in alternatives)
        for alternatives in case["required_concepts"]
    )

    forbidden_ok = not any(
        normalize_text(term) in normalized_answer for term in case["forbidden_terms"]
    )

    has_url = bool(re.search(r"https?://\S+", answer))
    has_page = bool(
        re.search(
            r"第\s*\d+\s*页|页码\s*[:：]?\s*\d+|p\.?\s*\d+",
            answer,
            flags=re.IGNORECASE,
        )
    )

    source_ok = not case["require_url"] or has_url
    page_ok = not case["require_page"] or has_page

    return {
        "concepts": concepts_ok,
        "forbidden": forbidden_ok,
        "source": source_ok,
        "page": page_ok,
    }


async def main() -> None:
    token = os.getenv("TRAVEL_AGENT_EVAL_TOKEN")

    if not token:
        raise RuntimeError("TRAVEL_AGENT_EVAL_TOKEN is not configured")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    passed_count = 0
    concept_passed = 0
    source_passed = 0
    source_required = 0
    page_passed = 0
    page_required = 0
    total_latency_ms = 0.0

    headers = {
        "Authorization": f"Bearer {token}",
    }

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers=headers,
        timeout=180,
    ) as client:
        for case in cases:
            start = perf_counter()

            try:
                response = await client.post(
                    "/api/v1/chat",
                    json={
                        "message": case["message"],
                    },
                )
                response.raise_for_status()
                answer = response.json()["reply"]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                latency_ms = (perf_counter() - start) * 1000
                total_latency_ms += latency_ms

                print(f"[ERROR] {case['id']} | latency={latency_ms:.0f}ms | {exc}")
                continue

            latency_ms = (perf_counter() - start) * 1000
            total_latency_ms += latency_ms

            checks = evaluate_answer(answer, case)
            passed = all(checks.values())

            passed_count += int(passed)
            concept_passed += int(checks["concepts"])

            if case["require_url"]:
                source_required += 1
                source_passed += int(checks["source"])

            if case["require_page"]:
                page_required += 1
                page_passed += int(checks["page"])

            status = "PASS" if passed else "FAIL"
            failed_checks = [name for name, result in checks.items() if not result]

            print(
                f"[{status}] {case['id']} | "
                f"behavior={case['expected_behavior']} | "
                f"latency={latency_ms:.0f}ms"
            )

            if failed_checks:
                print("  未通过：", failed_checks)
                print("  回答：", answer)

    case_count = len(cases)

    print()
    print(f"用例数量：{case_count}")
    print(f"整体通过率：{passed_count / case_count:.2%}")
    print(f"概念命中率：{concept_passed / case_count:.2%}")

    if source_required:
        print(f"来源标注率：{source_passed / source_required:.2%}")

    if page_required:
        print(f"页码标注率：{page_passed / page_required:.2%}")

    print(f"平均延迟：{total_latency_ms / case_count:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
