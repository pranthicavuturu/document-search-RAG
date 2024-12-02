import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

# Streamlit UI setup
st.set_page_config(page_title="Document Search with RAG", layout="wide")

st.title("Document Search with RAG")
st.markdown("Search academic papers and generate AI-powered responses.")

# Sidebar parameters
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter Search Query", "")
top_k = st.sidebar.slider("Number of Results", 1, 10, 5)
field = st.sidebar.selectbox("Search Field", ["all", "title", "abstract", "context"])

if st.sidebar.button("Search"):
    with st.spinner("Retrieving results and generating response..."):
        try:
            # Payload for the API
            payload = {"query": query, "top_k": top_k, "field": field}
            response = requests.post(f"{API_URL}/rag/", json=payload)

            if response.status_code == 200:
                # Extract data from the response
                data = response.json()
                generated_response = data.get("generated_response", "No response generated.")
                documents = data.get("documents", [])

                # Display AI-generated response
                st.subheader("AI-Generated Response")
                st.markdown(f"**Generated Response:**\n\n{generated_response}")

                # Display relevant documents
                st.subheader(f"Search Results (Field: {data['field']})")
                if documents:
                    for doc in documents:
                        st.markdown(
                            f"**Field**: {doc['field']}  \n"
                            f"**Title**: {doc['title']}  \n"
                            f"**Abstract**: {doc['abstract']}  \n"
                            f"**PDF File**: {doc['pdf_filename']}  \n"
                            f"**Relevance Score**: {doc['distance']:.4f}  \n"
                            f"---"
                        )
                else:
                    st.warning("No documents found.")
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) and [FastAPI](https://fastapi.tiangolo.com).")
