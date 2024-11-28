import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Search with RAG", layout="wide")

st.title("Document Search with RAG")
st.markdown("Search academic papers and get AI-generated insights.")

st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter Search Query", "")
top_k = st.sidebar.slider("Number of Results", 1, 10, 5)

if st.sidebar.button("Search"):
    with st.spinner("Retrieving and generating response..."):
        try:
            # Send request to the backend
            payload = {"query": query, "top_k": top_k}
            response = requests.post(f"{API_URL}/rag/", json=payload)

            if response.status_code == 200:
                data = response.json()
                generated_response = data.get("generated_response", "No response generated.")
                documents = data.get("documents", [])

                # Display generated response
                st.subheader("Generated Response")
                st.markdown(generated_response)

                # Display relevant documents
                st.subheader("Relevant Documents")
                for doc in documents:
                    st.markdown(
                        f"**Title**: {doc['title']}  \n"
                        f"**Abstract**: {doc['abstract']}  \n"
                        f"**PDF File**: {doc['pdf_filename']}  \n"
                        f"**Relevance Score**: {doc['distance']:.4f}  \n"
                        f"---"
                    )
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")

st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io), [FastAPI](https://fastapi.tiangolo.com), and GPT.")
