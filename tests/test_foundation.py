import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import enterprise_rag
from enterprise_rag.config import Settings


class FoundationTests(unittest.TestCase):
    def test_package_import_and_environment_loading(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "APP_ENV=test\nDOCUMENT_DIR=sample-documents\nLOG_LEVEL=debug\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_file)

        self.assertTrue(enterprise_rag.__doc__)
        self.assertEqual(settings.app_env, "test")
        self.assertEqual(settings.document_dir, Path("sample-documents"))
        self.assertEqual(settings.extracted_dir, Path("data/extracted"))
        self.assertEqual(settings.chunks_dir, Path("data/chunks"))
        self.assertEqual(settings.chroma_db_path, Path("chroma_db"))
        self.assertEqual(
            settings.chroma_collection_name, "enterprise_financial_chunks"
        )
        self.assertEqual(
            settings.embedding_model_name,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(settings.embedding_batch_size, 16)
        self.assertEqual(settings.semantic_top_k, 5)
        self.assertEqual(settings.chunk_max_chars, 1600)
        self.assertEqual(settings.chunk_overlap_chars, 200)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_process_environment_takes_precedence(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("APP_ENV=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"APP_ENV": "from-process"}, clear=True):
                settings = Settings.from_env(env_file)

        self.assertEqual(settings.app_env, "from-process")
