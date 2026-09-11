from typing import Any

import pytest

from app.rag.document_loader import Document
from app.rag.vector_store import (
    delete_documents_not_in,
    get_all_documents,
    upsert_documents,
)


class FakeCollection:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def upsert(self, **kwargs: Any) -> None:
        self.payload = kwargs


def make_document() -> Document:
    return Document(
        content="澳门博物馆相关旅行资料",
        metadata={
            "chunk_id": "guide-page-1-chunk-0",
            "document_id": "guide",
            "title": "澳门旅行指南",
            "publisher": "示例机构",
            "source_url": "https://example.com/guide",
            "page_number": 1,
            "chunk_index": 0,
        },
    )


def test_upsert_documents_writes_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(
        "app.rag.vector_store.get_collection",
        lambda: collection,
    )

    count = upsert_documents(
        [make_document()],
        [[0.1, 0.2, 0.3]],
    )

    assert count == 1
    assert collection.payload is not None
    assert collection.payload["ids"] == ["guide-page-1-chunk-0"]
    assert collection.payload["documents"] == ["澳门博物馆相关旅行资料"]


def test_upsert_documents_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        upsert_documents(
            [make_document()],
            [],
        )


def test_get_all_documents_reads_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadableCollection:
        def get(self, *, include: list[str]) -> dict:
            assert include == ["documents", "metadatas"]
            return {
                "documents": ["澳门历史城区资料"],
                "metadatas": [
                    {
                        "document_id": "macau-guide",
                        "chunk_id": "macau-guide-page-1-chunk-0",
                    }
                ],
            }

    monkeypatch.setattr(
        "app.rag.vector_store.get_collection",
        lambda: ReadableCollection(),
    )

    documents = get_all_documents()

    assert documents[0].content == "澳门历史城区资料"
    assert documents[0].metadata["document_id"] == "macau-guide"


def test_delete_documents_not_in_removes_stale_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeletableCollection:
        deleted_ids: list[str] = []

        def get(self, *, include: list[str]) -> dict:
            assert include == []
            return {"ids": ["current-chunk", "stale-chunk"]}

        def delete(self, *, ids: list[str]) -> None:
            self.deleted_ids = ids

    collection = DeletableCollection()
    monkeypatch.setattr(
        "app.rag.vector_store.get_collection",
        lambda: collection,
    )

    count = delete_documents_not_in({"current-chunk"})

    assert count == 1
    assert collection.deleted_ids == ["stale-chunk"]
