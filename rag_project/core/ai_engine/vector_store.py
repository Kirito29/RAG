import os
import faiss
import numpy as np
from openai import OpenAI
from django.conf import settings

from .document_reader import DocumentReader


class VectorStore:

    def __init__(self):

        self.client = OpenAI()

        self.index_path = os.path.join(settings.BASE_DIR, "vector_store.index")

        self.dimension = 1536

        self.index = faiss.IndexFlatL2(self.dimension)

        reader = DocumentReader()
        self.documents = reader.load_documents()


    def create_embeddings(self, texts):

        embeddings = []

        for text in texts:

            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )

            embedding = response.data[0].embedding

            embeddings.append(embedding)

        return embeddings


    def build_index(self):

        reader = DocumentReader()

        documents = reader.load_documents()

        self.documents = documents

        embeddings = self.create_embeddings(documents)

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        print("DOCUMENTS:", documents)
        print("TOTAL DOCS:", len(documents))
        faiss.write_index(self.index, self.index_path)


    def load_index(self):

        if os.path.exists(self.index_path):

            self.index = faiss.read_index(self.index_path)


    def search(self, query, k=3):

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

