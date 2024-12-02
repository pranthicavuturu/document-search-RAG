import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Paths
EMBEDDINGS_DIR = "../embeddings/embeddings_generated"
FAISS_TITLE_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_title_index.index")
FAISS_CONTEXT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_context_index.index")
FAISS_CHUNK_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_chunk_index.index")
METADATA_MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "paper_metadata.json")
CHUNK_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json")

# Load FAISS indices
title_index = faiss.read_index(FAISS_TITLE_INDEX_PATH)
context_index = faiss.read_index(FAISS_CONTEXT_INDEX_PATH)
chunk_index = faiss.read_index(FAISS_CHUNK_INDEX_PATH)

# Load metadata with UTF-8 encoding
with open(METADATA_MAPPING_PATH, 'r', encoding='utf-8') as f:
    paper_metadata = json.load(f)

with open(CHUNK_METADATA_PATH, 'r', encoding='utf-8') as f:
    chunk_metadata = json.load(f)

# Load SentenceTransformer model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load T5 model for answer generation
t5_model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def generate_query_embedding(query):
    return model.encode(query, show_progress_bar=False).reshape(1, -1)

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs.input_ids, max_length=150, num_beams=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

app = FastAPI()

class QueryModel(BaseModel):
    query: str
    filter: str = Query("title", regex="^(title|context|chunk)$")

@app.get('/search')
def search(query: str, filter: str = Query("title")):
    query_embedding = generate_query_embedding(query)

    index = {
        "title": title_index,
        "context": context_index,
        "chunk": chunk_index
    }.get(filter, title_index)

    distances, indices = index.search(query_embedding, k=10)

    results = []
    contexts = []

    if filter == "chunk":
        chunk_embeddings_flat = np.load(os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy"))
        for i, distance in zip(indices[0], distances[0]):
            try:
                chunk_info = chunk_metadata[i]
                doc_id = chunk_info["doc_id"]
                start_idx = chunk_info["start_idx"]
                end_idx = chunk_info["end_idx"]
                chunk_context = " ".join(chunk_embeddings_flat[start_idx:end_idx])
                relevance_score = 1 / (1 + distance)  # Transform distance to a relevance score
                results.append({
                    "title": paper_metadata[doc_id]["title"],
                    "relevance_score": relevance_score
                })
                contexts.append(chunk_context)
            except IndexError:
                print(f"IndexError: Skipping index {i} which is out of range for chunk_metadata or paper_metadata")
    else:
        for i, distance in zip(indices[0], distances[0]):
            try:
                result = paper_metadata[i]
                relevance_score = 1 / (1 + distance)  # Transform distance to a relevance score
                results.append({
                    "title": result["title"],
                    "relevance_score": relevance_score
                })
                contexts.append(result[filter])
            except IndexError:
                print(f"IndexError: Skipping index {i} which is out of range for paper_metadata")

    # Concatenate contexts for answer generation
    combined_context = " ".join(contexts)
    answer = generate_answer(query, combined_context)

    return {"results": results, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
