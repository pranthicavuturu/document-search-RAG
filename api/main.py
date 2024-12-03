import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

EMBEDDINGS_DIR = "../embeddings/embeddings_generated"
FAISS_TITLE_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "title_faiss_index.idx")
FAISS_CONTEXT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "context_faiss_index.idx")
FAISS_CHUNK_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_faiss_index.idx")
METADATA_MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "paper_metadata.json")
CHUNK_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json")
LINKS_PATH = os.path.join(EMBEDDINGS_DIR, "links.json")

# Load pre-generated FAISS indices for similarity search
title_index = faiss.read_index(FAISS_TITLE_INDEX_PATH)
context_index = faiss.read_index(FAISS_CONTEXT_INDEX_PATH)
chunk_index = faiss.read_index(FAISS_CHUNK_INDEX_PATH)

# Load metadata mappings for papers and chunks
with open(METADATA_MAPPING_PATH, 'r', encoding='utf-8') as f:
    paper_metadata = json.load(f)

with open(CHUNK_METADATA_PATH, 'r', encoding='utf-8') as f:
    chunk_metadata = json.load(f)

# Load the JSON file to create a mapping of titles to links
with open(LINKS_PATH, 'r', encoding='utf-8') as file:
    link_data = json.load(file)

# Create a dictionary mapping titles to links
title_to_link = {item['title']: item['link'] for item in link_data}

# Initialize a SentenceTransformer model for generating embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load the T5 model and tokenizer for generating answers based on context
t5_model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def generate_query_embedding(query):
    """
    Generate a query embedding for similarity search.
    """
    return model.encode(query, show_progress_bar=False).reshape(1, -1)

def generate_answer(question, context):
    """
    Generate an answer to a given question based on the provided context.
    """
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs.input_ids, max_length=150, num_beams=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

app = FastAPI()

class QueryModel(BaseModel):
    """
    Define the data model for the query request.
    """
    query: str
    filter: str = Query("title", regex="^(title|context|chunk)$")

@app.get('/search')
def search(query: str, filter: str = Query("title")):
    """
    Perform a similarity search using FAISS and generate an answer.
    """
    # Generate query embedding
    query_embedding = generate_query_embedding(query)

    # Select appropriate FAISS index based on the filter
    index = {
        "title": title_index,
        "context": context_index,
        "chunk": chunk_index
    }.get(filter, title_index)

    distances, indices = index.search(query_embedding, k=10)

    results = []
    contexts = []
    embeddings = []

    if filter == "chunk":
        for i, distance in zip(indices[0], distances[0]):
            try:
                chunk_info = chunk_metadata[i]
                doc_id = chunk_info["doc_id"]
                chunk_texts = chunk_info["chunks"]
                chunk_context = " ".join(chunk_texts)
                relevance_score = 1 / (1 + distance)
                title = paper_metadata[doc_id]["title"]
                link = title_to_link.get(title, "Link not available")
                results.append({
                    "title": title,
                    "relevance_score": relevance_score,
                    "link": link
                })
                contexts.append(chunk_context)
                # Generate embedding for the title
                title_embedding = model.encode(title).tolist()
                embeddings.append(title_embedding)
            except (IndexError, KeyError) as e:
                print(f"Error retrieving chunk metadata: {e}")
    else:
        for i, distance in zip(indices[0], distances[0]):
            try:
                result = paper_metadata[i]
                relevance_score = 1 / (1 + distance)
                title = result["title"]
                link = title_to_link.get(title, "Link not available")
                results.append({
                    "title": title,
                    "relevance_score": relevance_score,
                    "link": link
                })
                contexts.append(result.get(filter, ""))
                # Generate embedding for the title
                title_embedding = model.encode(title).tolist()
                embeddings.append(title_embedding)
            except (IndexError, KeyError) as e:
                print(f"Error retrieving paper metadata: {e}")

    combined_context = " ".join(contexts)
    answer = generate_answer(query, combined_context)

    return {"results": results, "answer": answer, "embeddings": embeddings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
