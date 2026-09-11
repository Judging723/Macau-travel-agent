import json
from pathlib import Path

import pytest

from app.rag.document_loader import (
    has_useful_text,
    load_document,
    load_documents,
    remove_markdown_document_header,
)

SOURCE = {
    "document_id": "example-guide",
    "title": "示例旅行指南",
    "publisher": "示例机构",
    "source_url": "https://example.com/guide",
}


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


def test_loads_text_document(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "\n 澳门概览 \n\n 历史城区介绍 \n",
        encoding="utf-8",
    )

    documents = load_document(path, SOURCE)

    assert documents[0].content == "澳门概览\n历史城区介绍"
    assert documents[0].metadata["page_number"] == 1


def test_removes_markdown_metadata_header() -> None:
    text = """# 澳门世界遗产

来源：澳门特别行政区政府旅游局
来源网址：https://example.com/heritage
资料抓取日期：2026-09-08

## 妈阁庙
妈阁庙是澳门重要的历史建筑。
"""

    cleaned = remove_markdown_document_header(text)

    assert cleaned == "## 妈阁庙\n妈阁庙是澳门重要的历史建筑。"


def test_loads_pdf_by_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: str) -> None:
            self.pages = [
                FakePage("第一页包含足够多的正文内容"),
                FakePage(""),
                FakePage("第三页同样包含足够多的正文内容"),
            ]

    monkeypatch.setattr(
        "app.rag.document_loader.PdfReader",
        FakeReader,
    )
    monkeypatch.setattr(
        "app.rag.document_loader.extract_pdf_pages_with_ocr",
        lambda _path, page_indexes: {page_indexes[0]: "第二页由OCR识别得到的正文内容"},
    )
    path = tmp_path / "guide.pdf"
    path.touch()

    documents = load_document(path, SOURCE)

    assert len(documents) == 3
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2
    assert documents[1].content == "第二页由OCR识别得到的正文内容"
    assert documents[2].metadata["page_number"] == 3


def test_pdf_uses_ocr_when_text_layer_only_contains_page_number(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("12")]

    monkeypatch.setattr(
        "app.rag.document_loader.PdfReader",
        FakeReader,
    )
    monkeypatch.setattr(
        "app.rag.document_loader.extract_pdf_pages_with_ocr",
        lambda _path, page_indexes: {page_indexes[0]: "澳门旅游指南OCR识别正文内容"},
    )
    path = tmp_path / "scanned-guide.pdf"
    path.touch()

    documents = load_document(path, SOURCE)

    assert documents[0].content == "澳门旅游指南OCR识别正文内容"
    assert documents[0].metadata["page_number"] == 1


def test_useful_text_ignores_page_numbers() -> None:
    assert not has_useful_text("01 02 03")
    assert has_useful_text("澳门历史城区包含多处重要的世界文化遗产建筑")


def test_loads_documents_from_manifest(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.txt").write_text(
        "澳门旅行资料",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "example-guide",
                        "file_name": "guide.txt",
                        "title": "示例旅行指南",
                        "publisher": "示例机构",
                        "source_url": "https://example.com/guide",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = load_documents(
        manifest_path,
        knowledge_dir,
    )

    assert len(documents) == 1
    assert documents[0].content == "澳门旅行资料"
    assert documents[0].metadata["title"] == "示例旅行指南"


def test_rejects_empty_text_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(" \n ", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="contains no content",
    ):
        load_document(path, SOURCE)
