
import  csv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from rag import config



def get_store(embedder=None):
    embedder = embedder or OllamaEmbeddings(model=config.EMBED_MODEL, **config.EMBED_KWARGS)
    return Chroma(collection_name=config.COLLECTION_NAME,
                  embedding_function=embedder, persist_directory=config.DB_DIR)



def build_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = []

    with open(config.REGISTER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = config.CORPUS_DIR / row["file_name"]

            if not path.exists():
                print(f"WARNING: missing file {path}, skipped")
                continue

            docs = PyPDFLoader(str(path)).load()

            for doc in docs:
                if not doc.page_content.strip():
                    continue

                chunks.extend(splitter.split_documents([doc]))

    return chunks



def main(rebuild=False):
    store = get_store()
    if rebuild:
        store.delete_collection()
        store = get_store()

    if store._collection.count() == 0:
        chunks = build_chunks()
        store.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks in Chroma.")
    else:
        print(" Chroma database loaded:", store._collection.count(), "chunks")


if __name__ == "__main__":
    main()