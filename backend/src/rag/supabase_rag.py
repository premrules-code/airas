"""RAG system with Supabase + pgvector."""

import logging
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Document, Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from sqlalchemy import make_url

from config.settings import get_settings
from .metadata_extractor import SECMetadataTransform

logger = logging.getLogger(__name__)


class SupabaseRAG:
    """RAG system using Supabase + pgvector."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = None
        self.pipeline = None
        self.index = None
        self.query_engine = None

    def setup_database(self, perform_setup: bool = True):
        """Connect to Supabase and create ingestion pipeline."""
        logger.info("Setting up Supabase...")

        url = make_url(self.settings.postgres_connection_string)

        self.vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port or 5432,
            user=url.username,
            table_name="airas_documents",
            embed_dim=1536,
            perform_setup=perform_setup,
        )

        embed_model = LlamaSettings.embed_model

        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                ),
                SECMetadataTransform(),
                embed_model,
            ],
            vector_store=self.vector_store,
        )

        logger.info("Supabase ready")

    def build_index(self, documents: List[Document]):
        """Build index from documents via ingestion pipeline."""
        logger.info(f"Indexing {len(documents)} documents...")

        nodes = self.pipeline.run(documents=documents, show_progress=True)
        logger.info(f"Pipeline produced {len(nodes)} nodes")

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        logger.info("Index built")

    def load_index(self):
        """Load existing index."""
        logger.info("Loading index...")

        if self.vector_store is None:
            self.setup_database(perform_setup=False)

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        logger.info("Index loaded")

    def create_query_engine(self, top_k: int = 5):
        """Create query engine."""
        if self.index is None:
            self.load_index()

        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        self.query_engine = RetrieverQueryEngine(retriever=retriever)
        logger.info("Query engine ready")

    def query(self, query_text: str) -> str:
        """Query the system."""
        if self.query_engine is None:
            self.create_query_engine()

        response = self.query_engine.query(query_text)
        return str(response)
