import sys
from langchain_ollama import ChatOllama
from tools.rag_tool import search_lending_docs

question = " ".join(sys.argv[1:]) or "What credit score do I need for a personal loan?"

result = search_lending_docs(question, top_k=4)

if result["status"] != "ok":
    print(f"No answer: {result['status']} - {result['warnings']}")
    sys.exit()

matches = result["data"]["matches"]
context = "\n\n".join(m["text"] for m in matches)

llm = ChatOllama(model="qwen3.5:4b", reasoning=False, num_ctx=4096, num_gpu=0)

prompt = f"""Answer using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

print(llm.invoke(prompt).content)
