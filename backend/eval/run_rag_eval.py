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


def expand_query_variants(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同一事实点的不同问法展开为独立评测查询。"""

    expanded_cases: list[dict[str, Any]] = []

    for case in cases:
        queries = [case["query"], *case.get("query_variants", [])]

        for index, query in enumerate(queries, start=1):
            expanded_cases.append(
                {
                    **case,
                    "id": f"{case['id']}#{index}",
                    "query": query,
                }
            )

    return expanded_cases


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
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        help="送入重排模型的混合检索候选数，仅在 rerank 模式生效",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只输出未命中用例和汇总指标",
    )
    return parser.parse_args()


async def main(
    mode: str,
    rerank_candidates: int | None = None,
    summary_only: bool = False,
) -> None:
    fact_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = expand_query_variants(fact_cases)
    search_functions = {
        "dense": search_knowledge,
        "hybrid": search_hybrid_knowledge,
        "rerank": search_reranked_knowledge,
    }
    search = search_functions[mode]

    if rerank_candidates is not None and rerank_candidates <= 0:
        raise ValueError("rerank_candidates must be positive")
    if mode != "rerank" and rerank_candidates is not None:
        raise ValueError("--rerank-candidates can only be used with --mode rerank")

    init_vector_store(allow_in_memory_fallback=False)

    # 预热一次，不让首次连接时间影响正式延迟
    if mode == "rerank":
        await search(
            "旅行",
            top_k=1,
            candidate_top_k=rerank_candidates,
        )
    else:
        await search("旅行", top_k=1)

    print(f"检索模式：{mode}")
    if mode == "rerank":
        print(f"重排候选数：{rerank_candidates or '默认配置'}")
    print()
    print(f"事实点数量：{len(fact_cases)}")
    print(f"查询数量：{len(cases)}")
    print()

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0
    total_latency_ms = 0.0
    category_metrics: dict[str, dict[str, float]] = {}

    for case in cases:
        start = perf_counter()

        search_arguments = {
            "query": case["query"],
            "top_k": 5,
        }
        if mode == "rerank":
            search_arguments["candidate_top_k"] = rerank_candidates

        results = await search(**search_arguments)

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

        category = case.get("category", "standard")
        metrics = category_metrics.setdefault(
            category,
            {
                "count": 0,
                "hit_at_1": 0,
                "reciprocal_rank_sum": 0.0,
            },
        )
        metrics["count"] += 1
        metrics["hit_at_1"] += int(relevant_rank == 1)

        if relevant_rank is not None:
            metrics["reciprocal_rank_sum"] += 1 / relevant_rank

        rank_text = str(relevant_rank) if relevant_rank else "MISS"

        if not summary_only or relevant_rank is None:
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

    for category, metrics in category_metrics.items():
        count = metrics["count"]
        print(
            f"{category}：Hit@1={metrics['hit_at_1'] / count:.2%}，"
            f"MRR@5={metrics['reciprocal_rank_sum'] / count:.4f}，"
            f"查询数={int(count)}"
        )


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(
        main(
            arguments.mode,
            arguments.rerank_candidates,
            arguments.summary_only,
        )
    )
