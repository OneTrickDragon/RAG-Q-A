from ingest import extract_text_from_pdf
from embed_retrieve import RAGVectorStore
from generate import call_ollama

if __name__ == "__main__":
    print("📘 Ask Your Notes (Local RAG)")
    path = input("Enter path to your lecture notes (PDF): ").strip()

    print("Loading and chunking...")
    text = extract_text_from_pdf(path)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    store = RAGVectorStore()
    store.add_documents(chunks)
    print("✅ Documents embedded and indexed.")

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        top_chunks = store.query(q, k=5)
        context = "\n".join(top_chunks)
        answer = call_ollama(context, q)
        print("\n💡 Answer:\n", answer.strip())
