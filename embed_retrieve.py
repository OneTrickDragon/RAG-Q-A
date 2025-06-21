from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []

    def add_documents(self, chunks):
        print("🔗 Generating embeddings...")
        
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            print("✅ Embedding successful.")
        except Exception as e:
            print("❌ Embedding failed:", e)
            raise e

        print("📥 Storing documents and embeddings...")
        
        valid_chunks = []
        valid_embeddings = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if len(chunk) > 2000:
                print(f"⚠️ Skipping chunk {i+1} (too long): {len(chunk)} chars")
                continue
            
            if not chunk.strip():
                print(f"⚠️ Skipping empty chunk {i+1}")
                continue
            
            valid_chunks.append(chunk)
            valid_embeddings.append(embedding)
        
        self.documents.extend(valid_chunks)
        self.embeddings.extend(valid_embeddings)
        
        print(f"✅ Successfully stored {len(valid_chunks)} documents!")
        print(f"📊 Total documents in store: {len(self.documents)}")

    def query(self, question, k=5):
        if not self.documents:
            print("⚠️ No documents in store!")
            return []
            
        try:
            # Encode the question
            question_emb = self.model.encode([question])[0]
            
            # Calculate similarities
            similarities = cosine_similarity([question_emb], self.embeddings)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Return top documents
            top_docs = [self.documents[i] for i in top_indices]
            
            print(f"🔍 Found {len(top_docs)} relevant chunks")
            return top_docs
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []