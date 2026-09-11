import pytest

from app.rag.chunker import deduplicate_documents, split_documents
from app.rag.document_loader import Document


def make_document(
    content: str,
    page_number: int = 3,
) -> Document:
    return Document(
        content=content,
        metadata={
            "document_id": "example-guide",
            "title": "示例旅行指南",
            "publisher": "示例机构",
            "source_url": "https://example.com/guide.pdf",
            "page_number": page_number,
        },
    )


def test_short_document_becomes_one_chunk() -> None:
    chunks = split_documents(
        [make_document("澳门拥有丰富的历史文化景点。")],
        min_length=1,
    )

    assert len(chunks) == 1
    assert chunks[0].content == "澳门拥有丰富的历史文化景点。"
    assert chunks[0].metadata["chunk_id"] == ("example-guide-page-3-chunk-0")


def test_long_document_has_overlap() -> None:
    content = "".join(chr(0x4E00 + index) for index in range(1200))

    chunks = split_documents(
        [make_document(content)],
        chunk_size=500,
        overlap=100,
    )

    assert len(chunks) == 3
    assert chunks[0].content[-100:] == chunks[1].content[:100]


def test_filters_short_chunk() -> None:
    chunks = split_documents([make_document("澳门主要旅游地标")])

    assert chunks == []


def test_deduplicates_simplified_and_traditional_versions() -> None:
    repeated_content = "澳门博物馆展示澳门数百年来的历史变迁和多元文化。" * 3
    traditional_content = "澳門博物館展示澳門數百年來的歷史變遷和多元文化。" * 3
    simplified = make_document(
        f"简介：{repeated_content}",
    )
    traditional = make_document(
        f"詳細資料：{traditional_content}",
    )

    documents = deduplicate_documents([simplified, traditional])

    assert documents == [simplified]


def test_keeps_documents_with_different_facts() -> None:
    museum = make_document(
        "澳门博物馆位于大炮台，主要介绍澳门历史和多元文化。" * 3,
    )
    lighthouse = make_document(
        "东望洋灯塔建于1864年，是中国海岸第一座现代灯塔。" * 3,
    )

    documents = deduplicate_documents([museum, lighthouse])

    assert documents == [museum, lighthouse]


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "min_length"),
    [
        (0, 0, 1),
        (100, -1, 1),
        (100, 100, 1),
        (100, 10, 0),
    ],
)
def test_rejects_invalid_settings(
    chunk_size: int,
    overlap: int,
    min_length: int,
) -> None:
    with pytest.raises(ValueError):
        split_documents(
            [make_document("测试文本")],
            chunk_size=chunk_size,
            overlap=overlap,
            min_length=min_length,
        )
