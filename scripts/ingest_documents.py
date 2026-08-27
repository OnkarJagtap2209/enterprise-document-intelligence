"""Ingest one PDF into a structured Docling JSON artifact."""

import argparse
import logging
from pathlib import Path

from enterprise_rag.config import get_settings
from enterprise_rag.ingestion import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path, help="PDF file to ingest")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = IngestionPipeline(settings.extracted_dir).ingest(args.pdf_path)
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
