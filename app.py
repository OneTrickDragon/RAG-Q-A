from ingest import extract_text_from_pdf
from embed_retrieve import RAGVectorStore
from generate import call_ollama

if __name__ == "__main__":
    print("📘 Ask Your Notes (Local RAG)")
    path = input("Enter path to your lecture notes (PDF): ").strip()

    print("📥 Loading and chunking document...")
    try:
        text = extract_text_from_pdf(path)
        print(f"📄 Extracted text length: {len(text)} characters")
        if not text.strip():
            print("⚠️ The PDF contains no readable text. Is it scanned or image-based?")
            exit()
    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        exit()

    # Break into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"🧩 Created {len(chunks)} chunks.")

    if not chunks:
        print("⚠️ No chunks were created.")
        exit()

    store = RAGVectorStore()
    print("📦 Adding chunks to vector store...")
    store.add_documents(chunks)
    print("✅ Indexed and ready. You may now ask questions.")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            print("👋 Exiting.")
            break

        top_chunks = store.query(q, k=5)
        context = "\n".join(top_chunks)
        answer = call_ollama(context, q)
        print("\n💡 Answer:\n", answer.strip())
