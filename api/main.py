import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

app = FastAPI()

JSON_DIR = "../collected-data/arxiv/json"

EMBEDDINGS_DIR = "../embeddings/embeddings_generated"
FAISS_INDEX_PATH = "../embeddings/embeddings_generated/faiss_index.idx"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

try:
    with open(os.path.join(EMBEDDINGS_DIR, "paper_metadata.json"), "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
except Exception as e:
    raise RuntimeError(f"Error loading metadata: {e}")

try:
    index = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index: {e}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

@app.get("/")
def home():
    return {"message": "Welcome to the Paper Search API!"}

@app.get("/search/")
def search(query: str, top_k: int = 5):
    """
    Search for papers using FAISS and embeddings.

    Args:
    - query: User's search string.
    - top_k: Number of results to return.

    Returns:
    - List of the most relevant papers based on the query.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    try:
        query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)

        distances, indices = index.search(query_embedding, top_k)

        results = [
            {
                "title": metadata[idx]["title"],
                "abstract": metadata[idx].get("abstract", "No Abstract Available"),
                "pdf_filename": metadata[idx].get("pdf_filename", "No File Available"),
                "distance": float(dist),
            }
            for idx, dist in zip(indices[0], distances[0])
        ]

        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")
