import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pypdf import PdfReader

BACKEND_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BACKEND_DIR / "data" / "knowledge"
MANIFEST_PATH = BACKEND_DIR / "knowledge_manifest.json"

Metadata = dict[str, str | int | float | bool]
MIN_USEFUL_CHARACTERS = 10
OCR_DPI = 150
MARKDOWN_METADATA_PREFIXES = (
    "来源：",
    "来源网址：",
    "资料抓取日期：",
)


@dataclass(frozen=True)
class Document:
    """一段正文及其来源信息。"""

    content: str
    metadata: Metadata


def clean_text(text: str) -> str:
    """删除空行和每行首尾空白。"""

    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def remove_markdown_document_header(text: str) -> str:
    """删除已保存在 metadata 中的 Markdown 文档头。"""

    lines = text.splitlines()
    index = 0

    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines) and lines[index].strip().startswith("# "):
        index += 1

    while index < len(lines):
        line = lines[index].strip()

        if not line or line.startswith(MARKDOWN_METADATA_PREFIXES):
            index += 1
            continue

        break

    return "\n".join(lines[index:])


def has_useful_text(text: str) -> bool:
    """判断文字层是否包含正文，而不只是页码或符号。"""

    meaningful_characters = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    return len(meaningful_characters) >= MIN_USEFUL_CHARACTERS


@lru_cache(maxsize=1)
def get_ocr_engine() -> Any:
    """延迟创建 OCR 引擎，普通文本文件不会加载模型。"""

    from rapidocr import RapidOCR

    return RapidOCR()


def extract_pdf_pages_with_ocr(
    path: Path,
    page_indexes: list[int],
) -> dict[int, str]:
    """把指定 PDF 页面转为图片并进行 OCR。"""

    import pymupdf

    engine = get_ocr_engine()
    extracted: dict[int, str] = {}

    with pymupdf.open(path) as pdf:
        for page_index in page_indexes:
            pixmap = pdf[page_index].get_pixmap(
                dpi=OCR_DPI,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            result = engine(pixmap.tobytes("png"))
            text = clean_text("\n".join(result.txts or ()))

            if has_useful_text(text):
                extracted[page_index] = text

    return extracted


def load_pdf(
    path: Path,
    metadata: Metadata,
) -> list[Document]:
    """按页读取 PDF，保留真实页码。"""

    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")

    page_texts: dict[int, str] = {}
    pages_needing_ocr: list[int] = []

    for page_index, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")

        if has_useful_text(text):
            page_texts[page_index] = text
        else:
            pages_needing_ocr.append(page_index)

    if pages_needing_ocr:
        page_texts.update(
            extract_pdf_pages_with_ocr(
                path,
                pages_needing_ocr,
            )
        )

    documents = [
        Document(
            content=text,
            metadata={
                **metadata,
                "page_number": page_index + 1,
            },
        )
        for page_index, text in sorted(page_texts.items())
    ]

    if not documents:
        raise ValueError("PDF contains no extractable or OCR-readable text")

    return documents


def load_text(
    path: Path,
    metadata: Metadata,
) -> list[Document]:
    """读取 TXT 或 Markdown。"""

    raw_text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".md":
        raw_text = remove_markdown_document_header(raw_text)

    text = clean_text(raw_text)

    if not text:
        raise ValueError("Text document contains no content")

    return [
        Document(
            content=text,
            metadata={**metadata, "page_number": 1},
        )
    ]


def load_document(
    path: Path,
    metadata: Metadata,
) -> list[Document]:
    """根据扩展名选择文档解析器。"""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, metadata)
    if suffix in {".txt", ".md"}:
        return load_text(path, metadata)

    raise ValueError(f"Unsupported document format: {suffix}")


def load_documents(
    manifest_path: Path = MANIFEST_PATH,
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> list[Document]:
    """读取清单中的全部本地知识文档。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[Document] = []
    seen_ids: set[str] = set()

    for item in manifest["documents"]:
        document_id = item["id"]
        file_name = item["file_name"]

        if document_id in seen_ids:
            raise ValueError("document ids must be unique")
        if Path(file_name).name != file_name:
            raise ValueError("file_name must not contain a path")

        path = knowledge_dir / file_name

        if not path.is_file():
            raise ValueError(f"document file does not exist: {file_name}")

        seen_ids.add(document_id)
        metadata: Metadata = {
            "document_id": document_id,
            "title": item["title"],
            "publisher": item["publisher"],
            "source_url": item["source_url"],
        }

        for field in ("category", "published_at", "retrieved_at"):
            if value := item.get(field):
                metadata[field] = value

        documents.extend(load_document(path, metadata))

    return documents
