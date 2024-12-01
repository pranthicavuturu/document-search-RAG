import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
JSON_DIR = "../collected-data/arxiv/json"
EMBEDDINGS_DIR = "../embeddings/embeddings_generated"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Storage for embeddings and metadata
title_embeddings = []
context_embeddings = []
chunk_embeddings = []
metadata = []

def chunk_text(text, max_chunk_size=200):
    """
    Split the text into smaller chunks based on paragraphs or max character size.
    Args:
    - text: The body text to split.
    - max_chunk_size: The maximum number of characters per chunk.

    Returns:
    - List of text chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]  # Split by paragraphs
    chunks = []

    for para in paragraphs:
        if len(para) > max_chunk_size:
            # If a paragraph is too long, split it into smaller chunks
            for i in range(0, len(para), max_chunk_size):
                chunks.append(para[i:i + max_chunk_size])
        else:
            chunks.append(para)

    return chunks

def process_json_files(json_dir):
    """
    Process JSON files to extract titles, abstracts, and generate embeddings.

    Args:
    - json_dir: Path to the directory containing JSON files.

    Returns:
    - metadata: List of document metadata.
    - title_embeddings: List of title embeddings.
    - context_embeddings: List of abstract/introduction embeddings.
    - chunk_embeddings: List of embeddings for document chunks.
    """
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    # Load JSON content
                    paper_data = json.load(file)

                    # Extract title and abstract
                    title = paper_data.get("title", "").strip()
                    context = paper_data.get("context", "").strip()
                    body = paper_data.get("body", "").strip()

                    # Generate embeddings
                    if title:
                        title_embedding = model.encode(title, show_progress_bar=False)
                        title_embeddings.append(title_embedding)
                    else:
                        title_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))

                    if context:
                        context_embedding = model.encode(context, show_progress_bar=False)
                        context_embeddings.append(context_embedding)
                    else:
                        context_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))

                    # Generate body chunk embeddings
                    if body:
                        chunks = chunk_text(body)
                        chunk_embedding_list = model.encode(chunks, show_progress_bar=True)
                        chunk_embeddings.append(chunk_embedding_list)
                    else:
                        chunk_embeddings.append([])

                    # Save metadata
                    metadata.append({
                        "title": title,
                        "context": context,
                        "pdf_filename": paper_data.get("pdf_filename", ""),
                        "num_chunks": len(chunk_embeddings[-1]),  # Track the number of chunks
                    })


                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

    return metadata, title_embeddings, context_embeddings, chunk_embeddings


# Process JSON files and generate embeddings
print("Generating embeddings for titles and contexts...")
metadata, title_embeddings, context_embeddings, chunk_embeddings = process_json_files(JSON_DIR)

# Save embeddings and metadata
print("Saving embeddings and metadata...")
np.save(os.path.join(EMBEDDINGS_DIR, "title_embeddings.npy"), np.array(title_embeddings))
np.save(os.path.join(EMBEDDINGS_DIR, "context_embeddings.npy"), np.array(context_embeddings))

# Save chunk embeddings separately since they have varying lengths
chunk_embeddings_flat = [embedding for doc in chunk_embeddings for embedding in doc]
np.save(os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy"), np.array(chunk_embeddings_flat))

# Save chunk metadata for reconstruction
chunk_metadata = []
current_offset = 0
for doc_id, doc_chunks in enumerate(chunk_embeddings):
    chunk_metadata.append({
        "doc_id": doc_id,
        "start_idx": current_offset,
        "end_idx": current_offset + len(doc_chunks)
    })
    current_offset += len(doc_chunks)

with open(os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json"), "w", encoding="utf-8") as chunk_meta_file:
    json.dump(chunk_metadata, chunk_meta_file, indent=4, ensure_ascii=False)

with open(os.path.join(EMBEDDINGS_DIR, "paper_metadata.json"), "w", encoding="utf-8") as meta_file:
    json.dump(metadata, meta_file, indent=4, ensure_ascii=False)

print("Embedding generation complete. Files saved in 'embeddings/embeddings_generated/'.")
