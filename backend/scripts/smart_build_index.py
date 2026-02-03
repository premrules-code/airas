#!/usr/bin/env python3
"""
Smart index builder - only indexes NEW files.
Tracks which files are already indexed in Supabase to avoid duplicates.
Supports --force (rebuild all), --clear (wipe index), --file (index one file).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

from llama_index.core import Document

from config.settings import get_settings
from config.logging_config import setup_logging
from src.utils.llama_setup import configure_llama_index
from src.rag.supabase_rag import SupabaseRAG
from src.rag.metadata_extractor import SECMetadataExtractor
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

metadata_extractor = SECMetadataExtractor()


def get_indexed_files(connection_string: str) -> set:
    """Get list of already-indexed source files from Supabase."""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT metadata->>'source_file' as source_file
                FROM data_airas_documents
                WHERE metadata->>'source_file' IS NOT NULL
            """))
            indexed = {row[0] for row in result.fetchall()}
            logger.info(f"Already indexed: {len(indexed)} files")
            return indexed
    except Exception as e:
        logger.warning(f"Could not fetch indexed files: {e}")
        return set()


def clear_all_index(connection_string: str):
    """Clear entire index."""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM data_airas_documents"))
            conn.commit()
            logger.info("Cleared ALL documents from index")
    except Exception as e:
        logger.error(f"Error clearing index: {e}")


def clear_file_from_index(connection_string: str, filename: str):
    """Remove a specific file's chunks from the index."""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM data_airas_documents WHERE metadata->>'source_file' = :filename"),
                {"filename": filename}
            )
            conn.commit()
            logger.info(f"Cleared {result.rowcount} chunks for {filename}")
    except Exception as e:
        logger.error(f"Error clearing {filename}: {e}")


def make_document(filepath: Path) -> Document:
    """Read a file and create a Document with base metadata from filename."""
    content = filepath.read_text(encoding="utf-8")
    base_metadata = metadata_extractor.extract_from_filename(filepath.name)
    now = datetime.utcnow().isoformat()
    base_metadata["created_at"] = now
    base_metadata["updated_at"] = now
    return Document(text=content, metadata=base_metadata)


def main():
    parser = argparse.ArgumentParser(description="Smart AIRAS index builder (skips already-indexed files)")
    parser.add_argument("--force", action="store_true", help="Force rebuild all files")
    parser.add_argument("--clear", action="store_true", help="Clear entire index before building")
    parser.add_argument("--file", type=str, help="Index a specific file only")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SMART INDEX BUILDER")
    print("=" * 70 + "\n")

    # Setup
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_llama_index()

    conn_string = settings.postgres_connection_string

    # Clear if requested
    if args.clear:
        clear_all_index(conn_string)

    # Initialize RAG
    rag = SupabaseRAG()
    rag.setup_database()

    # Determine files to process
    raw_dir = settings.raw_dir

    if args.file:
        target = Path(args.file)
        if not target.exists():
            target = raw_dir / args.file
        if not target.exists():
            print(f"File not found: {args.file}")
            return
        all_files = [target]
    else:
        all_files = sorted(raw_dir.glob("*.txt"))

    if not all_files:
        print(f"No .txt files found in {raw_dir}")
        print("Run: python scripts/download_sec_filings.py --ticker AAPL")
        return

    print(f"Found {len(all_files)} .txt files in {raw_dir}")

    # Check which are already indexed
    if args.force or args.clear:
        new_files = all_files
        print(f"Force mode: will index all {len(new_files)} files")
    else:
        indexed_files = get_indexed_files(conn_string)
        new_files = [f for f in all_files if f.name not in indexed_files]
        skipped = len(all_files) - len(new_files)
        if skipped:
            print(f"Skipping {skipped} already-indexed files")

    if not new_files:
        print("\nAll files already indexed. Use --force to re-index.")
        return

    print(f"\nIndexing {len(new_files)} new files:")
    for f in new_files:
        print(f"  - {f.name}")

    # Create Documents with base metadata (pipeline handles chunking + enrichment)
    documents = []
    for filepath in new_files:
        if args.force and not args.clear:
            clear_file_from_index(conn_string, filepath.name)
        documents.append(make_document(filepath))

    print(f"\nCreated {len(documents)} documents")
    print("Building index (chunking, enriching metadata, generating embeddings)...")

    try:
        rag.build_index(documents)
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nIndexing failed: {e}")
        return

    print("\n" + "=" * 70)
    print("INDEX BUILT SUCCESSFULLY")
    print("=" * 70)
    print(f"  Files indexed: {len(new_files)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
