# Document Search Using RAG

## Overview
This project allows users to search academic papers using vector similarity search and embeddings.

## Features
- Backend API built with FastAPI.
- Frontend UI using Streamlit.
- FAISS for approximate nearest neighbor search.

## Setup Instructions

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
pip install fastapi uvicorn streamlit requests faiss-cpu sentence-transformers
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

### 5. Contributions

1.

## Troubleshooting

- **Backend not running?**
  Ensure FastAPI is correctly installed and `uvicorn` is running without errors.

- **Frontend not connecting?**
  Confirm that the backend is running at `http://127.0.0.1:8000` and that the `API_URL` in `ui/app.py` is correct.

---
