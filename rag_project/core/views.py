from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.db.models import Q
from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.http import JsonResponse

import requests

import os
import json
import faiss
import numpy as np

from django.conf import settings
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Directory containing the source documents
DOC_FOLDER = os.path.join(settings.BASE_DIR, "docs")

# Size of each text chunk
# (larger chunks = more context but worse retrieval precision)
CHUNK_SIZE = 500


def HomeView(request):
    """
    Build a FAISS vector index from local documents.

    Steps performed:
    1. Read .txt files from docs/ directory
    2. Break text into smaller chunks
    3. Generate embeddings for each chunk
    4. Store embeddings in a FAISS vector index
    5. Save metadata to map chunks back to source files

    Returns
    -------
    JsonResponse
        JSON response with indexing statistics.
    """

    texts = []      # stores chunked text
    metadata = []   # stores source information for each chunk

    # -----------------------------
    # Step 1: Load Documents
    # -----------------------------
    # Iterate through all files in docs directory
    for filename in os.listdir(DOC_FOLDER):

        # Only process .txt files
        if filename.endswith(".txt"):

            file_path = os.path.join(DOC_FOLDER, filename)

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # -----------------------------
            # Step 2: Chunk the Text
            # -----------------------------
            # Split long text into smaller segments
            # so embeddings represent manageable units
            for i in range(0, len(content), CHUNK_SIZE):

                chunk = content[i:i + CHUNK_SIZE]

                texts.append(chunk)

                # Store source file for traceability
                metadata.append({
                    "source": filename
                })

    # -----------------------------
    # Step 3: Generate Embeddings
    # -----------------------------
    embeddings = []

    for text in texts:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        # Extract embedding vector
        embeddings.append(response.data[0].embedding)

    # Convert to numpy array (required by FAISS)
    embeddings = np.array(embeddings).astype("float32")

    # -----------------------------
    # Step 4: Create FAISS Index
    # -----------------------------
    # FAISS needs the dimensionality of vectors
    dimension = embeddings.shape[1]

    # IndexFlatL2 uses Euclidean distance
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to index
    index.add(embeddings)

    # -----------------------------
    # Step 5: Save Index to Disk
    # -----------------------------
    index_path = os.path.join(settings.BASE_DIR, "vector.index")

    faiss.write_index(index, index_path)

    # -----------------------------
    # Step 6: Save Metadata
    # -----------------------------
    # Metadata maps chunks back to original documents
    metadata_path = os.path.join(settings.BASE_DIR, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "texts": texts,
            "metadata": metadata
        }, f)

    # -----------------------------
    # Response
    # -----------------------------
    return JsonResponse({
        "status": "success",
        "documents_processed": len(os.listdir(DOC_FOLDER)),
        "chunks_indexed": len(texts)
    })
    
class HomeView2(View):
    def get(self, request):
        return render(request, 'core/home.html')