import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Paper Search", layout="wide")

st.title("Document Search")
st.markdown("Search academic papers with an AI-powered system.")

st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter Search Query", "")
top_k = st.sidebar.slider("Number of Results", 1, 10, 5)

if st.sidebar.button("Search"):
    with st.spinner("Searching..."):
        try:
            # Send request to FastAPI backend
            response = requests.get(f"{API_URL}/search/", params={"query": query, "top_k": top_k})

            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write(f"**Results for query:** `{query}`")
                    for result in results:
                        st.markdown(
                            f"**Title**: {result['title']}  \n"
                            f"**Abstract**: {result['abstract']}  \n"
                            f"**PDF File**: {result['pdf_filename']}  \n"
                            f"**Relevance Score**: {result['distance']:.4f}  \n"
                            f"---"
                        )
                else:
                    st.warning("No results found!")
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) and [FastAPI](https://fastapi.tiangolo.com).")
