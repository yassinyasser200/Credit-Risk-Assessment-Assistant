
from rag import config
from rag.ingest import get_store



def search_lending_docs(query: str, top_k: int = config.DEFAULT_TOP_K) -> dict:

    if not isinstance(query, str) or not query.strip():
        return {"status": "error", "warnings": ["Empty query."]}

    try:
        store = get_store()

        if store._collection.count() == 0:
            return {"status": "error", "warnings": ["Vector store is empty"]}

        docs = store.as_retriever(
            search_kwargs={"k": max(1, int(top_k))}
        ).invoke(query)

        matches = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in docs
        ]

        if not matches:
            return {"status": "no_match", "warnings": ["No relevant passages found."]}

        return {
            "status": "ok",
            "data": {"query": query, "matches": matches},
            "warnings": []
        }

    except Exception as e:
        return {"status": "error", "warnings": [f"Search failed: {e}"]}
