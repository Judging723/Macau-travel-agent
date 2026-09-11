import asyncio

import pytest

from app.rag.document_loader import Document
from app.rag.rag_service import (
    RRF,
    ingest_knowledge_base,
    search_knowledge,
    search_reranked_knowledge,
)
from app.rag.reranker import RERANKER_MODEL, get_reranker, rerank_documents


def make_document(document_id: str, content: str) -> Document:
    return Document(
        content=content,
        metadata={
            "document_id": document_id,
            "title": "澳门旅行指南",
            "publisher": "示例机构",
            "source_url": "https://example.com/guide",
            "page_number": 1,
        },
    )


def test_ingest_knowledge_base_in_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed_calls: list[list[str]] = []

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(
        "app.rag.rag_service.load_documents",
        lambda: [
            make_document("one", "澳门博物馆介绍澳门历史和多元文化。" * 3),
            make_document("two", "路环拥有海岸步道和传统渔村景观。" * 4),
        ],
    )
    monkeypatch.setattr(
        "app.rag.rag_service.init_vector_store",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.rag.rag_service.embed_texts",
        fake_embed,
    )
    monkeypatch.setattr(
        "app.rag.rag_service.upsert_documents",
        lambda documents, _embeddings: len(documents),
    )
    deleted_ids: list[set[str]] = []
    monkeypatch.setattr(
        "app.rag.rag_service.delete_documents_not_in",
        lambda document_ids: deleted_ids.append(document_ids),
    )

    count = asyncio.run(ingest_knowledge_base(batch_size=1))

    assert count == 2
    assert len(embed_calls) == 2
    assert len(deleted_ids[0]) == 2


def test_search_knowledge_returns_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(_texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(
        "app.rag.rag_service.embed_texts",
        fake_embed,
    )
    monkeypatch.setattr(
        "app.rag.rag_service.search_documents",
        lambda _embedding, _top_k: [
            {
                "content": "澳门博物馆位于大炮台附近。",
                "distance": 0.1,
                "metadata": {
                    "title": "澳门旅行指南",
                    "publisher": "示例机构",
                    "source_url": "https://example.com/guide",
                    "page_number": 3,
                },
            }
        ],
    )

    results = asyncio.run(search_knowledge("澳门博物馆"))

    assert results[0]["title"] == "澳门旅行指南"
    assert results[0]["page_number"] == 3
    assert results[0]["score"] == 0.9


def test_rrf_rewards_result_found_by_both_retrievers() -> None:
    def make_result(chunk_id: str) -> dict:
        return {
            "content": chunk_id,
            "metadata": {"chunk_id": chunk_id},
        }

    dense_results = [
        make_result("dense-only"),
        make_result("found-by-both"),
    ]
    bm25_results = [
        make_result("found-by-both"),
        make_result("bm25-only"),
    ]

    results = RRF(
        [dense_results, bm25_results],
        top_k=3,
    )

    assert results[0]["metadata"]["chunk_id"] == "found-by-both"


def test_rerank_documents_orders_by_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_pairs: list[tuple[str, str]] = []

    class FakeReranker:
        def predict(
            self,
            pairs: list[tuple[str, str]],
            *,
            show_progress_bar: bool,
        ) -> list[float]:
            received_pairs.extend(pairs)
            return [0.1, 0.9]

    monkeypatch.setattr(
        "app.rag.reranker.get_reranker",
        lambda: FakeReranker(),
    )

    documents = [
        {
            "title": "天主教艺术博物馆",
            "content": "澳门天主教艺术博物馆介绍",
            "score": 0.03,
        },
        {
            "title": "澳门海事博物馆",
            "content": "澳门海事博物馆可以了解航海文化",
            "score": 0.02,
        },
    ]

    results = rerank_documents(
        query="澳门哪里可以了解航海文化？",
        documents=documents,
        top_k=2,
    )

    assert results[0]["content"] == "澳门海事博物馆可以了解航海文化"
    assert results[0]["score"] == 0.9
    assert received_pairs[0][1] == ("标题：天主教艺术博物馆\n正文：澳门天主教艺术博物馆介绍")


def test_get_reranker_uses_bge_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict = {}

    def fake_cross_encoder(**kwargs: object) -> object:
        arguments.update(kwargs)
        return object()

    get_reranker.cache_clear()
    monkeypatch.setattr(
        "app.rag.reranker.CrossEncoder",
        fake_cross_encoder,
    )
    monkeypatch.setattr(
        "app.rag.reranker.torch.backends.mps.is_available",
        lambda: False,
    )

    get_reranker()

    assert arguments["model_name_or_path"] == RERANKER_MODEL
    assert arguments["device"] == "cpu"
    assert "prompts" not in arguments
    get_reranker.cache_clear()


def test_search_reranked_knowledge_uses_hybrid_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_hybrid_search(
        query: str,
        top_k: int,
    ) -> list[dict]:
        assert query == "澳门土生葡人美食"
        assert top_k == 20

        return [
            {"content": "候选一"},
            {"content": "候选二"},
            {"content": "候选三"},
        ]

    def fake_rerank(
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        assert query == "澳门土生葡人美食"
        assert top_k == 2
        return documents[:top_k]

    monkeypatch.setattr(
        "app.rag.rag_service.search_hybrid_knowledge",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        "app.rag.rag_service.rerank_documents",
        fake_rerank,
    )

    results = asyncio.run(
        search_reranked_knowledge(
            "澳门土生葡人美食",
            top_k=2,
        )
    )

    assert len(results) == 2
