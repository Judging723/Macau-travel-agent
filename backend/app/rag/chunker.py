import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from opencc import OpenCC

from app.rag.document_loader import Document

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "",
]
SIMPLIFIER = OpenCC("t2s")
DEDUPLICATION_MIN_LENGTH = 80
DEDUPLICATION_NGRAM_SIZE = 5
DEDUPLICATION_THRESHOLD = 0.84


def normalize_content(text: str) -> str:
    """统一繁简体、大小写和标点，供重复判断使用。"""

    simplified = SIMPLIFIER.convert(text).lower()
    normalized = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", simplified))
    return re.sub(r"^(简介|详细资料)", "", normalized)


def _character_ngrams(text: str) -> set[str]:
    """把文本转换成连续五字符集合。"""

    return {
        text[index : index + DEDUPLICATION_NGRAM_SIZE]
        for index in range(len(text) - DEDUPLICATION_NGRAM_SIZE + 1)
    }


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """删除繁简体统一后完全相同或高度近似的长片段。"""

    unique_documents: list[Document] = []
    normalized_contents: list[str] = []
    ngram_sets: list[set[str]] = []

    for document in documents:
        normalized = normalize_content(document.content)

        if normalized in normalized_contents:
            continue

        ngrams = _character_ngrams(normalized)
        is_duplicate = False

        if len(normalized) >= DEDUPLICATION_MIN_LENGTH:
            for kept_content, kept_ngrams in zip(
                normalized_contents,
                ngram_sets,
                strict=True,
            ):
                if len(kept_content) < DEDUPLICATION_MIN_LENGTH:
                    continue

                similarity = len(ngrams & kept_ngrams) / len(ngrams | kept_ngrams)

                if similarity >= DEDUPLICATION_THRESHOLD:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_documents.append(document)
            normalized_contents.append(normalized)
            ngram_sets.append(ngrams)

    return unique_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 700,
    overlap: int = 100,
    min_length: int = 50,
) -> list[Document]:
    """切分文档，并过滤过短的片段。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")
    if min_length <= 0:
        raise ValueError("min_length must be positive")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        keep_separator="end",
    )
    chunks: list[Document] = []

    for document in documents:
        texts = [
            text
            for text in splitter.split_text(document.content)
            if len(text.strip()) >= min_length
        ]

        for index, text in enumerate(texts):
            metadata = {
                **document.metadata,
                "chunk_index": index,
            }
            metadata["chunk_id"] = (
                f"{metadata['document_id']}-page-{metadata['page_number']}-chunk-{index}"
            )
            chunks.append(
                Document(
                    content=text,
                    metadata=metadata,
                )
            )

    return chunks
