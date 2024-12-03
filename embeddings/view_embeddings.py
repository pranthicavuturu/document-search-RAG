import numpy as np
import json

TITLE_EMBEDDINGS_PATH = "../embeddings/embeddings_generated/title_embeddings.npy"
CONTEXT_EMBEDDINGS_PATH = "../embeddings/embeddings_generated/context_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "../embeddings/embeddings_generated/chunk_embeddings.npy"
METADATA_PATH = "../embeddings/embeddings_generated/paper_metadata.json"
CHUNK_METADATA_PATH = "../embeddings/embeddings_generated/chunk_metadata.json"

title_embeddings = np.load(TITLE_EMBEDDINGS_PATH)
context_embeddings = np.load(CONTEXT_EMBEDDINGS_PATH)
chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)

with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as chunk_meta_file:
    chunk_metadata = json.load(chunk_meta_file)

print(f"Total title embeddings: {len(title_embeddings)}")
print(f"Title embedding dimensions: {title_embeddings[0].shape}")
print(f"Total context embeddings: {len(context_embeddings)}")
print(f"Context embedding dimensions: {context_embeddings[0].shape}")
print(f"Total chunk embeddings: {len(chunk_embeddings)}")
print(f"Chunk embedding dimensions: {chunk_embeddings[0].shape}")

print("\n--- Title and Context Embeddings ---")
for i in range(min(5, len(title_embeddings))):  # Limit to first 5 for brevity
    print(f"\nDocument {i + 1}:")
    print(f"Title: {metadata[i]['title']}")
    print(f"Filename: {metadata[i]['pdf_filename']}")
    print(f"Title Embedding (first 10 dimensions): {title_embeddings[i][:10]}")
    print(f"Context Embedding (first 10 dimensions): {context_embeddings[i][:10]}")

print("\n--- Chunk Embeddings ---")
for i, chunk_meta in enumerate(chunk_metadata[:3]):  # Show chunks for first 3 documents
    print(f"\nDocument {chunk_meta['doc_id'] + 1}:")
    print(f"Title: {metadata[chunk_meta['doc_id']]['title']}")
    print(f"Number of Chunks: {chunk_meta['end_idx'] - chunk_meta['start_idx']}")

    for j in range(chunk_meta['start_idx'], chunk_meta['end_idx']):
        print(f"  Chunk {j - chunk_meta['start_idx'] + 1} Embedding (first 10 dimensions): {chunk_embeddings[j][:10]}")
