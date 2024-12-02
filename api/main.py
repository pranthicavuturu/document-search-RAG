import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import openai

app = FastAPI()

# Paths
EMBEDDINGS_DIR = "../embeddings/embeddings_generated"
TITLE_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "title_faiss_index.idx")
ABSTRACT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_faiss_index.idx")
CONTEXT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "context_faiss_index.idx")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "paper_metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load metadata
try:
    with open(METADATA_PATH, "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
except Exception as e:
    raise RuntimeError(f"Error loading metadata: {e}")

# Load FAISS indexes
try:
    title_index = faiss.read_index(TITLE_INDEX_PATH)
    abstract_index = faiss.read_index(ABSTRACT_INDEX_PATH)
    context_index = faiss.read_index(CONTEXT_INDEX_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index: {e}")

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Set OpenAI API Key (ensure this is set in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

# Pydantic model for search requests
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5  # Default number of results
    field: str = "all"  # Options: "title", "abstract", "context", "all"


@app.get("/")
def home():
    return {"message": "Welcome to the Paper Search API with RAG!"}


@app.post("/rag/")
def rag(request: SearchRequest):
    print("rag request")
    """
    Retrieval-Augmented Generation (RAG) endpoint.
    1. Search for relevant documents.
    2. Generate a response using GPT based on the query and retrieved documents.
    """
    query = request.query
    top_k = request.top_k
    field = request.field

    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    try:
        # Step 1: Retrieve relevant documents
        query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)

        # Perform FAISS search based on the selected field
        if field == "title":
            distances, indices = title_index.search(query_embedding, top_k)
            retrieved_docs = _format_results(indices, distances, metadata, "title")
        elif field == "abstract":
            distances, indices = abstract_index.search(query_embedding, top_k)
            retrieved_docs = _format_results(indices, distances, metadata, "abstract")
        elif field == "context":
            distances, indices = context_index.search(query_embedding, top_k)
            retrieved_docs = _format_results(indices, distances, metadata, "context")
        elif field == "all":
            # Search across all fields
            title_distances, title_indices = title_index.search(query_embedding, top_k)
            abstract_distances, abstract_indices = abstract_index.search(query_embedding, top_k)
            context_distances, context_indices = context_index.search(query_embedding, top_k)

            retrieved_docs = _combine_results(
                title_indices, title_distances,
                abstract_indices, abstract_distances,
                context_indices, context_distances,
                metadata
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid field specified.")

        # Step 2: Construct context for GPT
        context = "\n\n".join(
            [f"Title: {doc['title']}\nAbstract: {doc['abstract']}" for doc in retrieved_docs]
        )
        messages = [
            {"role": "system", "content": "You are an expert in document summarization and retrieval."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nGenerate a concise and informative response."},
        ]

        # Step 3: Generate response using GPT
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with "gpt-3.5-turbo" if needed
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        generated_response = response["choices"][0]["message"]["content"].strip()

        return {"query": query, "generated_response": generated_response, "documents": retrieved_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG process: {e}")


def _format_results(indices, distances, metadata, field_name):
    """
    Format results for a single field.
    """
    return [
        {
            "field": field_name,
            "title": metadata[idx]["title"],
            "abstract": metadata[idx].get("abstract", "No Abstract Available"),
            "pdf_filename": metadata[idx].get("pdf_filename", "No File Available"),
            "distance": float(dist),
        }
        for idx, dist in zip(indices[0], distances[0])
    ]


def _combine_results(title_indices, title_distances, abstract_indices, abstract_distances,
                     context_indices, context_distances, metadata):
    """
    Combine results from multiple fields.
    """
    combined_results = []

    def append_results(indices, distances, field_name):
        for idx, dist in zip(indices[0], distances[0]):
            combined_results.append({
                "field": field_name,
                "title": metadata[idx]["title"],
                "abstract": metadata[idx].get("abstract", "No Abstract Available"),
                "pdf_filename": metadata[idx].get("pdf_filename", "No File Available"),
                "distance": float(dist),
            })

    append_results(title_indices, title_distances, "title")
    append_results(abstract_indices, abstract_distances, "abstract")
    append_results(context_indices, context_distances, "context")

    # Sort by distance (relevance)
    combined_results = sorted(combined_results, key=lambda x: x["distance"])

    return combined_results[:10]  # Return top 10 combined results
