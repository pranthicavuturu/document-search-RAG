## Setup Instructions

pip install streamlit

uvicorn api.main:app --reload
streamlit run ui/app.py

Open http://localhost:8501 in your browser to see the Streamlit frontend.
The frontend communicates with the FastAPI backend at http://127.0.0.1:8000.
