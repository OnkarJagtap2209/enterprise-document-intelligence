"""Structure-aware chunking for persisted Docling extraction artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from enterprise_rag.models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)

_HEADING_LABELS = {"section_header", "title"}
_IGNORED_TEXT_LABELS = {"page_header", "page_footer"}


class ChunkingError(ValueError):
    """Raised when an extraction artifact cannot produce valid chunks."""


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    output_path: Path
    chunks: tuple[DocumentChunk, ...]
    text_chunk_count: int
    table_chunk_count: int


@dataclass(slots=True)
class _ChunkDraft:
    content: str
    pages: tuple[int, ...]
    section_context: tuple[str, ...]
    content_type: str
    source_refs: tuple[str, ...]
    structural_labels: tuple[str, ...]
    provenance: tuple[dict[str, Any], ...]
    structured_content: dict[str, Any] | None = None


@dataclass(slots=True)
class _TextBuffer:
    section_context: tuple[str, ...]
    pages: tuple[int, ...]
    texts: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)


class StructureAwareChunker:
    """Build coherent text and table chunks from Docling's item hierarchy."""

    def __init__(self, max_chars: int = 1600, overlap_chars: int = 200) -> None:
        if max_chars <= 0:
            raise ChunkingError("max_chars must be greater than zero")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ChunkingError("overlap_chars must be between zero and max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, artifact: dict[str, Any]) -> tuple[DocumentChunk, ...]:
        document_id, source_filename, document = _validate_extraction_artifact(
            artifact
        )
        drafts: list[_ChunkDraft] = []
        headings: dict[int, str] = {}
        buffer: _TextBuffer | None = None

        def flush_buffer() -> None:
            nonlocal buffer
            if buffer is None:
                return
            drafts.append(
                _ChunkDraft(
                    content=_with_section_context(
                        buffer.section_context, "\n\n".join(buffer.texts)
                    ),
                    pages=buffer.pages,
                    section_context=buffer.section_context,
                    content_type="text",
                    source_refs=tuple(buffer.source_refs),
                    structural_labels=tuple(dict.fromkeys(buffer.labels)),
                    provenance=tuple(buffer.provenance),
                )
            )
            buffer = None

        for item, source_ref, level, collection in _iter_ordered_items(document):
            for deeper_level in tuple(headings):
                if deeper_level > level:
                    headings.pop(deeper_level)

            label = str(item.get("label", ""))
            text = str(item.get("text", "")).strip()
            if collection == "texts" and label in _HEADING_LABELS and text:
                flush_buffer()
                headings[level] = text
                continue

            section_context = tuple(headings[key] for key in sorted(headings))
            if collection == "tables":
                flush_buffer()
                drafts.extend(
                    self._chunk_table(item, source_ref, section_context)
                )
                continue

            if (
                collection != "texts"
                or not text
                or label in _IGNORED_TEXT_LABELS
            ):
                continue

            pages = _pages_from_provenance(item.get("prov"))
            provenance = _provenance(item.get("prov"))
            prefix_length = len(_section_prefix(section_context))
            body_limit = max(1, self.max_chars - prefix_length)

            if len(text) > body_limit:
                flush_buffer()
                overlap = min(self.overlap_chars, max(0, body_limit - 1))
                for fragment in _split_oversized(text, body_limit, overlap):
                    drafts.append(
                        _ChunkDraft(
                            content=_with_section_context(section_context, fragment),
                            pages=pages,
                            section_context=section_context,
                            content_type="text",
                            source_refs=(source_ref,),
                            structural_labels=(label,),
                            provenance=provenance,
                        )
                    )
                continue

            if buffer is not None:
                combined_body = "\n\n".join((*buffer.texts, text))
                same_context = buffer.section_context == section_context
                same_pages = buffer.pages == pages
                if (
                    not same_context
                    or not same_pages
                    or len(_with_section_context(section_context, combined_body))
                    > self.max_chars
                ):
                    flush_buffer()

            if buffer is None:
                buffer = _TextBuffer(section_context=section_context, pages=pages)
            buffer.texts.append(text)
            buffer.source_refs.append(source_ref)
            buffer.labels.append(label)
            buffer.provenance.extend(provenance)

        flush_buffer()
        if not drafts:
            raise ChunkingError("Extraction artifact produced no chunkable content")
        return _finalize_chunks(drafts, document_id, source_filename)

    def _chunk_table(
        self,
        table: dict[str, Any],
        source_ref: str,
        section_context: tuple[str, ...],
    ) -> list[_ChunkDraft]:
        data = table.get("data")
        if not isinstance(data, dict):
            raise ChunkingError(f"Table {source_ref} has no structured data")

        rows = data.get("grid")
        if not isinstance(rows, list) or not rows:
            cells = data.get("table_cells")
            rows = [cells] if isinstance(cells, list) and cells else []
        if not rows:
            raise ChunkingError(f"Table {source_ref} contains no rows or cells")

        row_lines = [_render_table_row(row) for row in rows]
        row_count = int(data.get("num_rows", len(rows)))
        column_count = int(data.get("num_cols", 0))
        title = f"Table {row_count}x{column_count}"
        prefix_length = len(_section_prefix(section_context)) + len(title) + 2
        body_limit = max(1, self.max_chars - prefix_length)
        provenance = _provenance(table.get("prov"))
        pages = _pages_from_provenance(table.get("prov"))
        drafts: list[_ChunkDraft] = []
        start = 0

        while start < len(rows):
            end = start
            selected_lines: list[str] = []
            while end < len(rows):
                candidate = "\n".join((*selected_lines, row_lines[end]))
                if selected_lines and len(candidate) > body_limit:
                    break
                if not selected_lines and len(candidate) > body_limit:
                    overlap = min(self.overlap_chars, max(0, body_limit - 1))
                    for fragment in _split_oversized(candidate, body_limit, overlap):
                        drafts.append(
                            self._table_draft(
                                title,
                                fragment,
                                table,
                                source_ref,
                                section_context,
                                pages,
                                provenance,
                                rows,
                                start,
                                start + 1,
                            )
                        )
                    end += 1
                    selected_lines = []
                    break
                selected_lines.append(row_lines[end])
                end += 1

            if selected_lines:
                drafts.append(
                    self._table_draft(
                        title,
                        "\n".join(selected_lines),
                        table,
                        source_ref,
                        section_context,
                        pages,
                        provenance,
                        rows,
                        start,
                        end,
                    )
                )
            start = end

        return drafts

    @staticmethod
    def _table_draft(
        title: str,
        rendered_rows: str,
        table: dict[str, Any],
        source_ref: str,
        section_context: tuple[str, ...],
        pages: tuple[int, ...],
        provenance: tuple[dict[str, Any], ...],
        rows: list[Any],
        row_start: int,
        row_end: int,
    ) -> _ChunkDraft:
        data = table["data"]
        body = f"{title}, rows {row_start + 1}-{row_end}\n{rendered_rows}"
        return _ChunkDraft(
            content=_with_section_context(section_context, body),
            pages=pages,
            section_context=section_context,
            content_type="table",
            source_refs=(source_ref,),
            structural_labels=("table",),
            provenance=provenance,
            structured_content={
                "table_ref": source_ref,
                "num_rows": data.get("num_rows"),
                "num_cols": data.get("num_cols"),
                "row_start": row_start,
                "row_end": row_end,
                "rows": rows[row_start:row_end],
            },
        )


class ChunkingPipeline:
    """Load one extraction artifact, chunk it, and persist the result."""

    def __init__(self, output_dir: str | Path, chunker: StructureAwareChunker) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.chunker = chunker

    def run(self, extraction_path: str | Path) -> ChunkingResult:
        source_path, artifact = load_extraction_artifact(extraction_path)
        chunks = self.chunker.chunk(artifact)
        document_id = artifact["document_id"]
        source_filename = artifact["source"]["filename"]
        output_path = self.output_dir / (
            f"{Path(source_filename).stem}.{document_id[:12]}.chunks.json"
        )
        payload = {
            "artifact_schema": "enterprise_rag.chunks.v1",
            "source_artifact": str(source_path),
            "document_id": document_id,
            "source": artifact["source"],
            "chunking": {
                "strategy": "structure_aware_v1",
                "max_chars": self.chunker.max_chars,
                "overlap_chars": self.chunker.overlap_chars,
                "chunk_count": len(chunks),
            },
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            raise ChunkingError(f"Could not persist chunk artifact: {exc}") from exc

        text_count = sum(
            chunk.metadata.content_type == "text" for chunk in chunks
        )
        table_count = sum(
            chunk.metadata.content_type == "table" for chunk in chunks
        )
        logger.info(
            "Chunking completed: chunks=%d text=%d tables=%d output=%s",
            len(chunks),
            text_count,
            table_count,
            output_path,
        )
        return ChunkingResult(
            output_path=output_path.resolve(),
            chunks=chunks,
            text_chunk_count=text_count,
            table_chunk_count=table_count,
        )


def load_extraction_artifact(path: str | Path) -> tuple[Path, dict[str, Any]]:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise ChunkingError(f"Extraction artifact does not exist: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChunkingError(f"Could not load extraction artifact: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ChunkingError("Extraction artifact must contain a JSON object")
    _validate_extraction_artifact(artifact)
    return artifact_path.resolve(), artifact


def _validate_extraction_artifact(
    artifact: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if artifact.get("artifact_schema") != "enterprise_rag.extraction.v1":
        raise ChunkingError("Unsupported or missing extraction artifact schema")
    document_id = artifact.get("document_id")
    source = artifact.get("source")
    document = artifact.get("document")
    if not isinstance(document_id, str) or not document_id:
        raise ChunkingError("Extraction artifact is missing document_id")
    if not isinstance(source, dict) or not isinstance(source.get("filename"), str):
        raise ChunkingError("Extraction artifact is missing source filename")
    if not isinstance(document, dict) or not document.get("body"):
        raise ChunkingError("Extraction artifact is missing the Docling document body")
    return document_id, source["filename"], document


def _iter_ordered_items(
    document: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], str, int, str]]:
    collections = {
        name: document.get(name, [])
        for name in ("texts", "tables", "pictures", "groups")
    }

    def visit(reference: Any, level: int) -> Iterator[tuple[dict, str, int, str]]:
        if not isinstance(reference, dict) or not isinstance(reference.get("$ref"), str):
            return
        source_ref = reference["$ref"]
        parts = source_ref.removeprefix("#/").split("/")
        if len(parts) != 2 or parts[0] not in collections:
            return
        collection, index_text = parts
        try:
            item = collections[collection][int(index_text)]
        except (IndexError, TypeError, ValueError):
            raise ChunkingError(f"Invalid Docling item reference: {source_ref}")
        if not isinstance(item, dict):
            raise ChunkingError(f"Malformed Docling item: {source_ref}")
        yield item, source_ref, level, collection
        for child in item.get("children", []):
            yield from visit(child, level + 1)

    body = document.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("children"), list):
        raise ChunkingError("Docling document body has no ordered children")
    for child in body["children"]:
        yield from visit(child, 1)


def _finalize_chunks(
    drafts: list[_ChunkDraft], document_id: str, source_filename: str
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    for index, draft in enumerate(drafts):
        identity = {
            "document_id": document_id,
            "chunk_index": index,
            "content_type": draft.content_type,
            "source_refs": draft.source_refs,
            "content": draft.content,
        }
        digest = sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        metadata = ChunkMetadata(
            chunk_id=f"chunk_{digest}",
            document_id=document_id,
            source_filename=source_filename,
            pages=draft.pages,
            section_context=draft.section_context,
            content_type=draft.content_type,
            chunk_index=index,
            source_refs=draft.source_refs,
            structural_labels=draft.structural_labels,
        )
        chunks.append(
            DocumentChunk(
                content=draft.content,
                metadata=metadata,
                provenance=draft.provenance,
                structured_content=draft.structured_content,
            )
        )
    return tuple(chunks)


def _section_prefix(section_context: tuple[str, ...]) -> str:
    return f"Section: {' > '.join(section_context)}\n\n" if section_context else ""


def _with_section_context(section_context: tuple[str, ...], body: str) -> str:
    return f"{_section_prefix(section_context)}{body}".strip()


def _provenance(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _pages_from_provenance(value: Any) -> tuple[int, ...]:
    pages = {
        item["page_no"]
        for item in _provenance(value)
        if isinstance(item.get("page_no"), int)
    }
    return tuple(sorted(pages))


def _render_table_row(row: Any) -> str:
    if not isinstance(row, list):
        return str(row).strip()
    values: list[str] = []
    seen_cells: set[tuple[Any, ...]] = set()
    for cell in row:
        if not isinstance(cell, dict):
            values.append(str(cell).strip())
            continue
        identity = (
            cell.get("start_row_offset_idx"),
            cell.get("end_row_offset_idx"),
            cell.get("start_col_offset_idx"),
            cell.get("end_col_offset_idx"),
        )
        text = str(cell.get("text", "")).replace("\n", " ").strip()
        values.append("" if identity in seen_cells else text)
        seen_cells.add(identity)
    return " | ".join(values)


def _split_oversized(text: str, limit: int, overlap: int) -> list[str]:
    fragments: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max(1, limit // 2), end + 1)
            if boundary > start:
                end = boundary
        fragment = text[start:end].strip()
        if fragment:
            fragments.append(fragment)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1
    return fragments
