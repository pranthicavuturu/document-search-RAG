import numpy as np
import json

embedding_output = "../embeddings/embeddings_generated/embeddings.npy"
metadata_output = "../embeddings/embeddings_generated/paper_metadata.json"

embeddings = np.load(embedding_output)

with open(metadata_output, "r") as meta_file:
    metadata = json.load(meta_file)

print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings[0].shape}")

for i in range(min(5, len(embeddings))):
    print(f"\nEmbedding {i + 1}:")
    print(f"Title: {metadata[i]['title']}")
    print(f"Filename: {metadata[i]['filename']}")
    print(f"Embedding Vector: {embeddings[i]}")
    print(f"Embedding (first 10 dimensions): {embeddings[i][:10]}")  # Display first 10 dimensions
