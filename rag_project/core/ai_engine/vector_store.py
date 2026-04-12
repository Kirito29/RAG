import os
import faiss
import numpy as np
from openai import OpenAI
from django.conf import settings
import pickle

from .document_reader import DocumentReader


def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

class VectorStore:

    def __init__(self):

        self.client = OpenAI()

        self.index_path = os.path.join(settings.BASE_DIR, "vector_store.index")

        self.dimension = 1536

        self.index = faiss.IndexFlatL2(self.dimension)

        self.documents = []


    def create_embeddings(self, documents):

    # documents is now a list of dicts with "text" and "source" keys
        non_empty_docs = [doc for doc in documents if doc["text"].strip()]

        if not non_empty_docs:
            print("No text found in any documents to embed.")
            return []

        # Add filename into the text for better retrieval
        for doc in non_empty_docs:
            doc["embedding_text"] = f"Filename: {doc['source']}\n{doc['text']}"
        
        texts = [doc["embedding_text"] for doc in non_empty_docs]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]

        return embeddings


    def build_index(self):

        reader = DocumentReader()
        documents = reader.load_documents()

        chunked_docs = []

        for doc in documents:
            text = f"Filename: {doc['source']}\n{doc['text']}"

            chunks = [text[i:i+500] for i in range(0, len(text), 500)]

            for chunk in chunks:
                if chunk.strip():
                    chunked_docs.append({
                        "text": chunk,
                        "source": doc["source"]
                    })

        self.documents = chunked_docs

        texts = [doc["text"] for doc in chunked_docs]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        vectors = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)

        # ✅ SAVE FAISS INDEX
        faiss.write_index(self.index, self.index_path)

        # 🔥 SAVE DOCUMENTS WITH PICKLE
        import pickle

        docs_path = os.path.join(settings.BASE_DIR, "documents.pkl")

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        
        print("DOCUMENTS LOADED:")
        for doc in documents:
            print(doc["source"], "=>", doc["text"][:100])

        
        
        print("TOTAL CHUNKS:", len(chunked_docs))

    def load_index(self):

        if os.path.exists(self.index_path):

            self.index = faiss.read_index(self.index_path)
            

            docs_path = os.path.join(settings.BASE_DIR, "documents.pkl")

            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)

    def search(self, query, k=3):

        self.load_index() 

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        query_vector = np.array([response.data[0].embedding]).astype("float32")

        distances, indexes = self.index.search(query_vector, k)

        results = []

        for i in indexes[0]:
            if i < len(self.documents):
                results.append(self.documents[i])

        return results


