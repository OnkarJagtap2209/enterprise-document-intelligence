"""Create structure-aware chunks from one Docling extraction artifact."""

import argparse
import logging
from pathlib import Path

from enterprise_rag.config import get_settings
from enterprise_rag.ingestion import ChunkingPipeline, StructureAwareChunker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction_artifact", type=Path)
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    chunker = StructureAwareChunker(
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    result = ChunkingPipeline(settings.chunks_dir, chunker).run(
        args.extraction_artifact
    )
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
