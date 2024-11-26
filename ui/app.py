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
        response = requests.get(f"{API_URL}/search/", params={"query": query, "top_k": top_k})
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                st.write(f"**Results for query:** `{query}`")
                for result in results:
                    st.markdown(
                        f"**Title**: {result['title']}  \n"
                        f"**Link**: {'<Link soon>'}  \n"
                        f"---"
                    )
            else:
                st.warning("No results found!")
        else:
            st.error("Failed to connect to the backend.")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) and [FastAPI](https://fastapi.tiangolo.com).")
