from openai import OpenAI
from .vector_store import VectorStore


class ChatEngine:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def ask(self, question):

        # Retrieve relevant docs from vector store
        docs = self.vector_store.search(question)

        if not docs:
            return "No relevant documents found."

        # Convert list of dicts to a single string for context
        context = "\n\n".join(
    [f"[Source: {doc['source']}]\n{doc['text']}" for doc in docs if doc['text'].strip()]
)

        #Call your GPT model
        response = self.vector_store.client.chat.completions.create(
            model="gpt-4",
            messages=[
        {
            "role": "system",
            "content": "Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know based on the documents.'"
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
        ]   

        )

        return response.choices[0].message.content


