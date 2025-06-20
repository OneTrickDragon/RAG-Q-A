from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import uuid

class RAGVectorStore:
    def __init__(self, collection_name="docs"):
        self.client = chromadb.Client()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, texts):
        embeddings = self.model.encode(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(documents=texts, ids=ids, embeddings=embeddings)

    def query(self, question, k=5):
        question_emb = self.model.encode([question])[0]
        result = self.collection.query(embedding=question_emb, n_results=k)
        return result['documents'][0]
