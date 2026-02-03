"""Test RAG system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from config.logging_config import setup_logging
from src.utils.llama_setup import configure_llama_index
from src.rag.supabase_rag import SupabaseRAG

def main():
    print("\n" + "="*70)
    print("🧪 TESTING RAG SYSTEM")
    print("="*70 + "\n")
    
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_llama_index()
    
    rag = SupabaseRAG()
    rag.setup_database()
    rag.load_index()
    rag.create_query_engine()
    
    queries = [
        "What was Apple's revenue in 2023?",
        "What is Apple's gross margin?",
        "How much cash does Apple have?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("─"*70)
        response = rag.query(query)
        print(f"Response: {response}\n")
    
    print("="*70)
    print("✅ RAG TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
