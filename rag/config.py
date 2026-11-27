from pathlib import Path

RAG_DIR = Path(__file__).parent
CORPUS_DIR = RAG_DIR / "corpus"
REGISTER_PATH = RAG_DIR / "source_register.csv"
DB_DIR = str(RAG_DIR / "chroma_store")

COLLECTION_NAME = "lending_docs"

# Embeddings
EMBED_MODEL = "nomic-embed-text"
EMBED_KWARGS = {"num_gpu": 0}

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
DEFAULT_TOP_K = 4
