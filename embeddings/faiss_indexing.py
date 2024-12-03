# This script creates a FAISS index for efficient similarity search using embeddings that have already been generated
# loads embeddings from a .npy file.
# initializes a FAISS index using L2 (Euclidean) distance.
# adds the embeddings to the index.
# saves the index to a file for later use.
import faiss
import numpy as np

embeddings_path = "../embeddings/embeddings_generated/embeddings.npy"
embeddings = np.load(embeddings_path)

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)

index.add(embeddings)
print(f"FAISS index contains {index.ntotal} vectors.")

faiss.write_index(index, "../embeddings/embeddings_generated/faiss_index.idx")
print("FAISS index saved.")
