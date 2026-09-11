import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from opencc import OpenCC

from app.rag.rag_service import (
    search_hybrid_knowledge,
    search_knowledge,
    search_reranked_knowledge,
)
from app.rag.vector_store import init_vector_store

CASES_PATH = Path(__file__).with_name("rag_cases.json")
TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def normalize_text(text: str) -> str:
    """统一繁简体，并消除 PDF 抽取产生的空白差异。"""

    simplified = TRADITIONAL_TO_SIMPLIFIED.convert(text)
    return "".join(simplified.lower().split())


def is_relevant_result(
    result: dict[str, Any],
    case: dict[str, Any],
) -> bool:
    """判断结果是否命中人工标注的答案片段。"""

    content = normalize_text(result["content"])

    return any(
        result["title"] == passage.get("source", case["expected_source"])
        and result["page_number"] == passage["page_number"]
        and all(normalize_text(keyword) in content for keyword in passage["keywords"])
        for passage in case["relevant_passages"]
    )


def parse_args() -> argparse.Namespace:
    """读取命令行中的检索模式。"""

    parser = argparse.ArgumentParser(
        description="评测旅行知识库的检索效果",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rerank"],
        default="rerank",
        help="dense=向量检索，hybrid=混合检索，rerank=混合检索+重排",
    )
    return parser.parse_args()


async def main(mode: str) -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    search_functions = {
        "dense": search_knowledge,
        "hybrid": search_hybrid_knowledge,
        "rerank": search_reranked_knowledge,
    }
    search = search_functions[mode]

    init_vector_store(allow_in_memory_fallback=False)

    # 预热一次，不让首次连接时间影响正式延迟
    await search("旅行", top_k=1)

    print(f"检索模式：{mode}")
    print()

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0
    total_latency_ms = 0.0

    for case in cases:
        start = perf_counter()

        results = await search(
            query=case["query"],
            top_k=5,
        )

        latency_ms = (perf_counter() - start) * 1000
        total_latency_ms += latency_ms

        relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if is_relevant_result(result, case)
            ),
            None,
        )

        hit_at_1 += int(relevant_rank == 1)
        hit_at_3 += int(relevant_rank is not None and relevant_rank <= 3)
        hit_at_5 += int(relevant_rank is not None)

        if relevant_rank is not None:
            reciprocal_rank_sum += 1 / relevant_rank

        rank_text = str(relevant_rank) if relevant_rank else "MISS"

        print(f"{case['id']} | rank={rank_text} | latency={latency_ms:.0f}ms")

        if relevant_rank is None:
            retrieved_pages = [f"{result['title']}:p{result['page_number']}" for result in results]
            print("  Top 5：", retrieved_pages)

    case_count = len(cases)

    print()
    print(f"用例数量：{case_count}")
    print(f"Hit@1：{hit_at_1 / case_count:.2%}")
    print(f"Hit@3：{hit_at_3 / case_count:.2%}")
    print(f"Hit@5：{hit_at_5 / case_count:.2%}")
    print(f"MRR@5：{reciprocal_rank_sum / case_count:.4f}")
    print(f"平均延迟：{total_latency_ms / case_count:.0f}ms")


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(main(arguments.mode))
