import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

json_folder = "../collected-data/arxiv/json"
embedding_output = "../embeddings/embeddings_generated/embeddings.npy"
metadata_output = "../embeddings/embeddings_generated/paper_metadata.json"

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = []
metadata = []

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):  # Process only JSON files
        with open(os.path.join(json_folder, filename), "r") as file:
            try:
                paper = json.load(file)

                text_to_embed = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('body', '')}"

                embedding = model.encode(text_to_embed)

                embeddings.append(embedding)
                metadata.append({
                    "filename": filename,
                    "title": paper.get("title", ""),
                    "pdf_filename": paper.get("pdf_filename", ""),
                })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

embeddings = np.array(embeddings)

os.makedirs(os.path.dirname(embedding_output), exist_ok=True)
np.save(embedding_output, embeddings)

with open(metadata_output, "w") as meta_file:
    json.dump(metadata, meta_file, indent=4)

print(f"Embeddings saved to {embedding_output}")
print(f"Metadata saved to {metadata_output}")
