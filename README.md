# document-search-RAG

## Overview
This project allows users to search academic papers using vector similarity search and embeddings.

## Features
- Backend API built with FastAPI.
- Frontend UI using Streamlit.
- FAISS for approximate nearest neighbor search.

## Setup Instructions

# Project Checklist: AI Research Paper Search Engine

## DATA

- [ ] **Document Collection**
  - [ ] Scrape papers from sources like AAAI, ICML, or arXiv.
  - [ ] Store raw files in a consistent format (PDFs, text files).
  - [ ] Extract text from raw files using libraries like PyPDF2 or pdfplumber.

- [ ] **Text Cleaning**
  - [ ] Normalize text (remove special characters, extra spaces, etc.).
  - [ ] Organize into structured JSON or text files with fields like title, abstract, and content.

## EMBEDDINGS

- [ ] **Creative Approach for Embedding Generation**
  - [ ] Use sentence-transformers to generate vector embeddings for documents.
  - [ ] Save embeddings as a `.npy` file and maintain metadata (e.g., document IDs).

## INDEX CREATION

- [ ] **Build Approximate Nearest Neighbor (ANN) Index**
  - [ ] Build an ANN index using FAISS.
  - [ ] Test index accuracy with example queries.
  - [ ] Prepare embeddings and index for use in the backend.

## BACKEND

- [ ] **API Development**
  - [ ] Build an API for searching documents using the index.
  - [ ] Define `/search/` endpoint: Accepts a query and returns top-k similar documents.
  - [ ] Define `/health/` endpoint: Checks if the API is running.
  - [ ] Connect the FAISS index to the API for real-time similarity search.
  - [ ] Handle missing queries or invalid inputs gracefully.

## FRONTEND

- [ ] **User Interface Creation**
  - [ ] Create input fields for search queries.
  - [ ] Display results (e.g., titles, abstracts) in a clean layout.
  - [ ] Fetch search results dynamically from the backend.
  - [ ] Handle loading states and errors.

## DEPLOYMENT

- [ ] **Backend and Frontend Hosting**
  - [ ] Deploy the FastAPI app using platforms like AWS, Render, or Heroku.
  - [ ] Deploy the Streamlit app using Streamlit Cloud or similar platforms.
  - [ ] Automate deployment with GitHub Actions or similar tools.
  - [ ] Store large files (e.g., embeddings) in cloud storage like AWS S3.
  - [ ] Should produce: Hosted backend and frontend URLs, CI/CD pipeline configuration.

## DOCUMENTATION AND EXPECTATIONS

- [ ] **Clear Documentation**
  - [ ] Provide clear documentation of functions.
  - [ ] Reasoning for all methods chosen.
  - [ ] Print out loss (where applicable).

## Add on's

- [ ] https://projector.tensorflow.org/


## STEPS TO LAUNCH THE APP


# Document Search System

This project provides a system to search academic papers using a FastAPI backend and a Streamlit frontend.

---

## Prerequisites

- Python 3.8 or later
- pip (Python package manager)

---

## Setup Instructions

### 1. Install Dependencies

Run the following command to install the required libraries:

```bash
pip install fastapi uvicorn streamlit requests
```

---

### 2. Start the Backend (FastAPI)

1. Navigate to the `api/` directory:
   ```bash
   cd api
   ```

2. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

3. The backend will be available at:
   ```
   http://127.0.0.1:8000
   ```

---

### 3. Start the Frontend (Streamlit)

1. Open a new terminal and navigate to the `ui/` directory:
   ```bash
   cd ui
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. The frontend will open in your browser at:
   ```
   http://localhost:8501
   ```

---

### 4. Use the System

1. Access the Streamlit app in your browser.
2. Enter a search query (e.g., "Deep Learning") and specify the number of results.
3. Click **Search** to fetch results from the FastAPI backend.

---

## Troubleshooting

- **Backend not running?**
  Ensure FastAPI is correctly installed and `uvicorn` is running without errors.

- **Frontend not connecting?**
  Confirm that the backend is running at `http://127.0.0.1:8000` and that the `API_URL` in `ui/app.py` is correct.

---
