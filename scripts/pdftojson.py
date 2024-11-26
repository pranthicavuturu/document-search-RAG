import os
import pdfplumber
import json
import re
# import pymupdf/ pdfminer

# Paths
pdf_dir = "../collected-data/arxiv/pdf"
output_dir = "../collected-data/arxiv/json"

os.makedirs(output_dir, exist_ok=True)

def extract_abstract_from_lines(lines):
    """
    Extract abstract from a list of lines.
    Stops when encountering a heading or keywords section.
    """
    abstract_lines = []
    recording = False

    for line in lines:
        # Start recording when "Abstract" is found
        if "abstract" in line.lower():
            recording = True
            continue

        # Stop recording at delimiters like section headings or keywords
        if recording and (re.match(r"^\d+\.\s+[A-Z]", line) or "keywords" in line.lower()):
            break
        
        if recording:
            abstract_lines.append(line.strip())

    return " ".join(abstract_lines).strip()


def extract_text_from_pdf(pdf_path):
    """
    Extract text, title, and abstract from a PDF file.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from the first page for title and abstract
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            # Handle cases where the first page has no text
            if not text:
                print(f"Warning: No text found on the first page of {pdf_path}")
                return None
            
            lines = text.split("\n")

            # Extract title (assume first line is the title)
            title = lines[0].strip() if lines else ""

            # Extract abstract
            abstract = extract_abstract_from_lines(lines)

            # Extract full text (all pages)
            full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            # Return the extracted components
            return {
                "title": title or "",  # Leave empty if no title is identified
                "abstract": abstract or "",  # Leave empty if no abstract is identified
                "body": full_text,
                "pdf_filename": os.path.basename(pdf_path)
            }
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def process_all_pdfs(pdf_directory, output_directory):
    """
    Process all PDFs in the directory and save as JSON files.
    """
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, file_name)
            print(f"Processing: {file_name}")
            paper_data = extract_text_from_pdf(pdf_path)
            
            if paper_data:
                # Save the extracted data as a JSON file
                json_file_name = file_name.replace(".pdf", ".json")
                output_path = os.path.join(output_directory, json_file_name)
                
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(paper_data, json_file, indent=4, ensure_ascii=False)

    print(f"Processing complete. JSON files saved in '{output_directory}'.")


# Run the script
process_all_pdfs(pdf_dir, output_dir)