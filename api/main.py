import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import openai

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

# Set OpenAI API Key
openai.api_key = ""

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message": "Welcome to the Paper Search API with RAG!"}

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


@app.post("/rag/")
def rag(request: RAGRequest):
    """
    RAG (Retrieval-Augmented Generation) endpoint.
    1. Search for relevant documents.
    2. Generate a response using GPT based on query and retrieved documents.

    Args:
    - query: User's query string.
    - top_k: Number of documents to retrieve for context.

    Returns:
    - Generated response and relevant documents.
    """
    query = request.query
    top_k = request.top_k

    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    try:
        query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)

        retrieved_docs = [
            {
                "title": metadata[idx]["title"],
                "abstract": metadata[idx].get("abstract", "No Abstract Available"),
                "pdf_filename": metadata[idx].get("pdf_filename", "No File Available"),
                "distance": float(dist),
            }
            for idx, dist in zip(indices[0], distances[0])
        ]

        context = "\n\n".join(
            [f"Title: {doc['title']}\nAbstract: {doc['abstract']}" for doc in retrieved_docs]
        )
        messages = [
            {"role": "system", "content": "You are an expert in document summarization and retrieval."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nGenerate a concise and informative response."},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with "gpt-3.5-turbo" if using GPT-3.5
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        generated_response = response["choices"][0]["message"]["content"].strip()

        return {"query": query, "generated_response": generated_response, "documents": retrieved_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG process: {e}")
