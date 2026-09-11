import logging
from typing import Any

import chromadb
import httpx

from app.config import get_settings
from app.rag.document_loader import Document

EMBEDDINGS_URL = "http://127.0.0.1:11434/api/embed"

logger = logging.getLogger("travel_agent.rag")
settings = get_settings()
_chroma_client = None


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """使用本地 Ollama 模型生成向量。"""

    if not texts:
        return []

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            EMBEDDINGS_URL,
            json={
                "model": settings.embedding_model,
                "input": texts,
            },
        )
        response.raise_for_status()

    return response.json()["embeddings"]


def init_vector_store(
    *,
    allow_in_memory_fallback: bool = True,
) -> None:
    """连接 ChromaDB；开发环境可退回内存模式。"""

    global _chroma_client

    try:
        _chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        _chroma_client.heartbeat()
    except Exception:
        if not allow_in_memory_fallback:
            raise

        logger.warning("ChromaDB server unavailable; using in-memory client")
        _chroma_client = chromadb.Client()


def get_collection():
    """获取旅行知识集合。"""

    global _chroma_client

    if _chroma_client is None:
        _chroma_client = chromadb.Client()

    return _chroma_client.get_or_create_collection(
        name=settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
    )


def upsert_documents(
    documents: list[Document],
    embeddings: list[list[float]],
) -> int:
    """把文档片段及其向量写入 ChromaDB。"""

    if not documents:
        return 0
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must have the same length")

    get_collection().upsert(
        ids=[str(document.metadata["chunk_id"]) for document in documents],
        documents=[document.content for document in documents],
        embeddings=embeddings,
        metadatas=[document.metadata for document in documents],
    )

    return len(documents)


def get_all_documents() -> list[Document]:
    """读取已入库的全部片段，供 BM25 建立关键词索引。"""

    results = get_collection().get(
        include=["documents", "metadatas"],
    )
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    return [
        Document(
            content=content,
            metadata=metadatas[index],
        )
        for index, content in enumerate(documents)
    ]


def delete_documents_not_in(document_ids: set[str]) -> int:
    """删除不属于本次知识库的旧片段。"""

    collection = get_collection()
    stored_ids = collection.get(include=[]).get("ids") or []
    stale_ids = [document_id for document_id in stored_ids if document_id not in document_ids]

    if stale_ids:
        collection.delete(ids=stale_ids)

    return len(stale_ids)


def search_documents(
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """使用查询向量搜索最相近的文档。"""

    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    return [
        {
            "content": content,
            "metadata": metadatas[0][index],
            "distance": distances[0][index],
        }
        for index, content in enumerate(documents[0])
    ]
