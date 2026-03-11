from openai import OpenAI
from .vector_store import VectorStore


class ChatEngine:

    def __init__(self):

        self.client = OpenAI()

        self.vector_store = VectorStore()

        self.vector_store.load_index()


    def ask(self, question):

        # search documents
        docs = self.vector_store.search(question)
        print("SEARCH RESULTS:", docs)
        context = "\n\n".join(docs)

        prompt = f"""
You are an AI assistant answering questions using the provided documents.

Documents:
{context}

Question:
{question}

Answer clearly using the information in the documents.
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You answer questions based on provided documents."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
