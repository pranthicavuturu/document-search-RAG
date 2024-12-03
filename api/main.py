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

# FAISS indexes we have generated and stored
title_index = faiss.read_index(FAISS_TITLE_INDEX_PATH)
context_index = faiss.read_index(FAISS_CONTEXT_INDEX_PATH)
chunk_index = faiss.read_index(FAISS_CHUNK_INDEX_PATH)

with open(METADATA_MAPPING_PATH, 'r', encoding='utf-8') as f:
    paper_metadata = json.load(f)

with open(CHUNK_METADATA_PATH, 'r', encoding='utf-8') as f:
    chunk_metadata = json.load(f)

with open(LINKS_PATH, 'r', encoding='utf-8') as file:
    link_data = json.load(file)

title_to_link = {item['title']: item['link'] for item in link_data}

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# load the T5 model and tokenizer for generating answers based on context
t5_model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def generate_query_embedding(query):
    """
    Generate an embedding for a given query using the SentenceTransformer model.

    Args:
        query (str): The query entered in the search box to generate embedding from.

    Returns:
        np.ndarray: A NumPy array representing the query embedding.
    """
    return model.encode(query, show_progress_bar=False).reshape(1, -1)

def generate_answer(question, context):
    """
    Generate an answer to a question (which is the input from the user) and
    based on the given context (which is the retrieved documents) using a T5 model.

    Args:
        question (str): The question to answer.
        context (str): The context to use for generating the answer.

    Returns:
        str: The generated answer.
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
    Perform a similarity search on the given query using FAISS indices and generate an answer.

    Args:
        query (str): The search query.
        filter (str): The filter type ('title', 'context', or 'abstract (chunk)').

    Returns:
        dict: A dictionary containing search results, a generated answer, and embeddings.
    """
    query_embedding = generate_query_embedding(query)

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
