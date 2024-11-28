import faiss
import numpy as np

embeddings_path = "../embeddings/embeddings_generated/embeddings.npy"
embeddings = np.load(embeddings_path)

embedding_dim = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)

index.add(embeddings)
print(f"FAISS index contains {index.ntotal} vectors.")

faiss.write_index(index, "../embeddings/embeddings_generated/faiss_index.idx")
print("FAISS index saved.")
