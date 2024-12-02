import streamlit as st
import requests

st.title("Document Search")

# Input fields for query and filter
query = st.text_input("Enter your query:")
filter = st.selectbox("Filter by:", ["title", "context", "chunk"])

if st.button("Search"):
    response = requests.get("http://localhost:8000/search", params={"query": query, "filter": filter})

    if response.status_code == 200:
        results = response.json().get("results", [])
        answer = response.json().get("answer", "")

        st.subheader("Generated Response")
        st.write(answer)

        if results:
            st.subheader("Search Results")
            for result in results:
                st.write(f"Title: {result['title']}")
                st.write(f"Relevance Score: {result['relevance_score']:.4f}")
        else:
            st.write("No results found.")
    else:
        st.write("Error: Unable to fetch results from the backend.")
