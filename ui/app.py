import streamlit as st
import requests
import json

# Load the JSON file to create a mapping of titles to links
link_file_path = "../embeddings/embeddings_generated/links.json"
with open(link_file_path, 'r') as file:
    link_data = json.load(file)

# Create a dictionary mapping titles to links
title_to_link = {item['title']: item['link'] for item in link_data}

st.title("Document Search")

# Input fields for query and filter
query = st.text_input("Enter your query:")
filter = st.selectbox("Filter by:", ["title", "context", "abstract"])

if st.button("Search"):
    # Make a request to the backend search endpoint
    response = requests.get("http://localhost:8000/search", params={"query": query, "filter": filter})

    if response.status_code == 200:
        # Parse the response
        results = response.json().get("results", [])
        answer = response.json().get("answer", "")

        # Display generated response
        st.subheader("Generated Response")
        st.write(answer)

        if results:
            st.subheader("Search Results")
            # Display the search results with titles and corresponding links
            for result in results:
                title = result['title']
                link = title_to_link.get(title, "Link not available")

                # Display title and link
                st.write(f"Title: {title}")
                st.write(f"Link: {link}")
                st.write("---")  # A line separator for better readability
        else:
            st.write("No results found.")
    else:
        st.write("Error: Unable to fetch results from the backend.")
