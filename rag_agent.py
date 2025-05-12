import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai

class RAGAgent:
    def __init__(self, data_dir: str, embedding_model: str = 'all-MiniLM-L6-v2', openai_api_key: str = None):
        self.data_dir = data_dir
        self.documents = []
        self.chunks = []
        self.chunk_sources = []
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.embeddings = None
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self._load_documents()
        self._embed_chunks()

    def _load_documents(self):
        for fname in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, fname)
            with open(path, 'r') as f:
                text = f.read()
                # Simple chunking: split by double newlines
                chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
                self.chunks.extend(chunks)
                self.chunk_sources.extend([fname]*len(chunks))
                self.documents.append((fname, text))

    def _embed_chunks(self):
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec, dtype=np.float32), top_k)
        return [(self.chunks[i], self.chunk_sources[i]) for i in I[0]]

    def generate_answer(self, query: str, top_k: int = 3, model: str = "gpt-3.5-turbo") -> str:
        """
        Retrieve relevant chunks and generate an answer using OpenAI LLM.
        """
        context_chunks = self.retrieve(query, top_k)
        context = "\n".join([chunk for chunk, _ in context_chunks])
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n"
            f"Context:\n{context}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not set.")
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
