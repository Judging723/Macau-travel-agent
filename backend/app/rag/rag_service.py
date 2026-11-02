import asyncio
import json
import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.rag.chunker import deduplicate_documents, split_documents
from app.rag.document_loader import Document, load_documents
from app.rag.reranker import rerank_documents
from app.rag.vector_store import (
    delete_documents_not_in,
    embed_texts,
    get_all_documents,
    init_vector_store,
    search_documents,
    upsert_documents,
)

settings = get_settings()
logger = logging.getLogger("travel_agent.rag")


def log_search_metrics(**metrics: Any) -> None:
    logger.info(
        json.dumps(
            {
                "event": "rag_search",
                **metrics,
            },
            ensure_ascii=False,
        )
    )


def tokenize(text: str) -> list[str]:
    """使用搜索引擎模式进行中文分词。"""

    return [token.strip().lower() for token in jieba.cut_for_search(text) if token.strip()]


@lru_cache(maxsize=1)
def get_bm25_index() -> tuple[list[Document], BM25Okapi]:
    """从 Chroma 读取已入库片段并建立 BM25 索引。"""

    documents = get_all_documents()

    tokenized_documents = [tokenize(document.content) for document in documents]

    return documents, BM25Okapi(tokenized_documents)


def search_bm25_documents(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """使用 BM25 检索知识库片段。"""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    documents, index = get_bm25_index()
    scores = index.get_scores(tokenize(query))

    ranked_indexes = sorted(
        range(len(documents)),
        key=lambda position: scores[position],
        reverse=True,
    )

    results: list[dict[str, Any]] = []

    for position in ranked_indexes:
        score = float(scores[position])

        if score <= 0:
            continue

        document = documents[position]

        results.append(
            {
                "content": document.content,
                "metadata": document.metadata,
                "bm25_score": score,
            }
        )

        if len(results) == top_k:
            break

    return results


def RRF(
    result_groups: list[list[dict[str, Any]]],
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """按照排名融合多组检索结果。"""
    scores: dict[str, Any] = {}
    documents: dict[str, dict[str, Any]] = {}

    for results in result_groups:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["metadata"]["chunk_id"]

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank_constant + rank)
            documents[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        {
            **documents[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids[:top_k]
    ]


async def ingest_knowledge_base(batch_size: int = 8) -> int:
    """加载、切分并写入全部本地知识文档。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks = deduplicate_documents(split_documents(load_documents()))

    if not chunks:
        return 0

    init_vector_store(allow_in_memory_fallback=False)
    inserted_count = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = await embed_texts([document.content for document in batch])
        inserted_count += upsert_documents(batch, embeddings)

    delete_documents_not_in({str(document.metadata["chunk_id"]) for document in chunks})

    get_bm25_index.cache_clear()

    return inserted_count


async def search_knowledge(
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """用自然语言搜索旅行知识。"""

    query_embedding = (await embed_texts([query]))[0]
    results = search_documents(
        query_embedding,
        top_k or settings.rag_top_k,
    )
    formatted_results: list[dict[str, Any]] = []

    for result in results:
        score = round(1 - result["distance"], 4)

        if score < settings.rag_score_threshold:
            continue

        metadata = result["metadata"]
        formatted_results.append(
            {
                "title": metadata["title"],
                "content": result["content"],
                "publisher": metadata["publisher"],
                "source_url": metadata["source_url"],
                "page_number": metadata["page_number"],
                "score": score,
                **{
                    field: metadata[field]
                    for field in ("category", "published_at", "retrieved_at")
                    if field in metadata
                },
            }
        )

    return formatted_results


async def search_hybrid_knowledge(
    query: str,
    top_k: int | None = None,
    candidate_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """融合向量检索和 BM25 检索结果。"""

    final_top_k = top_k or settings.rag_top_k
    retrieval_top_k = candidate_top_k or settings.rag_hybrid_candidate_k

    if final_top_k <= 0 or retrieval_top_k <= 0:
        raise ValueError("top_k and candidate_top_k must be positive")

    retrieval_top_k = max(final_top_k, retrieval_top_k)

    total_started = perf_counter()
    embedding_started = perf_counter()
    query_embedding = (await embed_texts([query]))[0]
    embedding_ms = (perf_counter() - embedding_started) * 1000

    dense_started = perf_counter()
    dense_results = search_documents(
        query_embedding,
        retrieval_top_k,
    )
    dense_ms = (perf_counter() - dense_started) * 1000

    bm25_started = perf_counter()
    bm25_results = search_bm25_documents(
        query,
        retrieval_top_k,
    )
    bm25_ms = (perf_counter() - bm25_started) * 1000

    fusion_started = perf_counter()
    fused_results = RRF(
        [dense_results, bm25_results],
        top_k=final_top_k,
    )
    fusion_ms = (perf_counter() - fusion_started) * 1000

    log_search_metrics(
        mode="hybrid",
        query_length=len(query),
        dense_candidates=len(dense_results),
        bm25_candidates=len(bm25_results),
        returned=len(fused_results),
        embedding_ms=round(embedding_ms, 2),
        dense_ms=round(dense_ms, 2),
        bm25_ms=round(bm25_ms, 2),
        fusion_ms=round(fusion_ms, 2),
        total_ms=round((perf_counter() - total_started) * 1000, 2),
    )

    return [
        {
            "title": result["metadata"]["title"],
            "content": result["content"],
            "publisher": result["metadata"]["publisher"],
            "source_url": result["metadata"]["source_url"],
            "page_number": result["metadata"]["page_number"],
            "score": round(result["rrf_score"], 6),
            **{
                field: result["metadata"][field]
                for field in ("category", "published_at", "retrieved_at")
                if field in result["metadata"]
            },
        }
        for result in fused_results
    ]


async def search_reranked_knowledge(
    query: str,
    top_k: int | None = None,
    candidate_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """对混合检索候选结果进行重排。"""

    final_top_k = top_k or settings.rag_top_k
    rerank_top_k = candidate_top_k or settings.rag_rerank_candidate_k

    if final_top_k <= 0 or rerank_top_k <= 0:
        raise ValueError("top_k and candidate_top_k must be positive")

    rerank_top_k = max(final_top_k, rerank_top_k)

    total_started = perf_counter()
    hybrid_started = perf_counter()
    candidates = await search_hybrid_knowledge(
        query=query,
        top_k=rerank_top_k,
        candidate_top_k=settings.rag_hybrid_candidate_k,
    )
    hybrid_ms = (perf_counter() - hybrid_started) * 1000

    rerank_started = perf_counter()
    results = await asyncio.to_thread(
        rerank_documents,
        query,
        candidates,
        final_top_k,
    )
    rerank_ms = (perf_counter() - rerank_started) * 1000

    log_search_metrics(
        mode="rerank",
        query_length=len(query),
        hybrid_candidates=len(candidates),
        returned=len(results),
        hybrid_ms=round(hybrid_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round((perf_counter() - total_started) * 1000, 2),
    )

    return results


if __name__ == "__main__":
    count = asyncio.run(ingest_knowledge_base())
    print(f"写入片段数：{count}")
