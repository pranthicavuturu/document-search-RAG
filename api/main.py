# File: api/main.py
# basic api just to get started
from fastapi import FastAPI, HTTPException

app = FastAPI()

papers = [
    {"id": 1, "title": "Deep Learning in AI", "abstract": "Explores deep learning techniques in AI."},
    {"id": 2, "title": "Natural Language Processing", "abstract": "An introduction to NLP methods."},
    {"id": 3, "title": "Computer Vision Trends", "abstract": "Discusses recent trends in computer vision."},
    {"id": 4, "title": "Reinforcement Learning", "abstract": "Basics of reinforcement learning and applications."},
]

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
        if query.lower() in paper["title"].lower() or query.lower() in paper["abstract"].lower()
    ]

    if not results:
        raise HTTPException(status_code=404, detail="No papers found matching your query.")

    return {"query": query, "results": results}
