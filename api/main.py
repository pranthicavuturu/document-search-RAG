import os
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Directory containing the JSON files
JSON_DIR = "../collected-data/arxiv/json"

# Load all papers from the JSON files
def load_papers():
    papers = []
    try:
        for file_name in os.listdir(JSON_DIR):
            if file_name.endswith(".json"):
                file_path = os.path.join(JSON_DIR, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    # Handle missing fields and malformed data gracefully
                    paper = {
                        "id": file_name,  # Use filename as ID for simplicity
                        "title": data.get("title", "No Title Available").strip(),
                        "abstract": data.get("abstract", "No Abstract Available").strip(),
                    }
                    papers.append(paper)
    except Exception as e:
        print(f"Error loading papers: {e}")
    return papers

papers = load_papers()

@app.get("/")
def home():
    return {"message": "Welcome to the Paper Search API!"}

@app.get("/search/")
def search(query: str):
    """
    Search for papers containing the query in their title or abstract.

    Args:
    - query: Search string from the user.

    Returns:
    - List of papers that match the query.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    # Simple search logic (case-insensitive)
    results = [
        paper for paper in papers
        if query.lower() in (paper["title"] or "").lower() or query.lower() in (paper["abstract"] or "").lower()
    ]

    if not results:
        raise HTTPException(status_code=404, detail="No papers found matching your query.")

    return {"query": query, "results": results}
