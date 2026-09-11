from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """加载并缓存重排模型。"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    return CrossEncoder(
        model_name_or_path=RERANKER_MODEL,
        device=device,
    )


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """根据问题与文档内容的相关性重新排序。"""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not documents:
        return []

    pairs = [
        (
            query,
            f"标题：{document['title']}\n正文：{document['content']}",
        )
        for document in documents
    ]

    scores = get_reranker().predict(
        pairs,
        show_progress_bar=False,
    )

    ranked_results = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {
            **document,
            "score": round(float(score), 4),
        }
        for document, score in ranked_results[:top_k]
    ]
