import os
import fitz  # PyMuPDF
import json
import re

# Paths
pdf_dir = "../collected-data/arxiv/pdf"
output_dir = "../collected-data/arxiv/json"

os.makedirs(output_dir, exist_ok=True)

def extract_context(lines):
    """
    Extract the 'Abstract' and 'Introduction' sections from a list of lines.
    Stops when encountering a heading after 'Introduction'.
    """
    context_lines = []
    recording_abstract = False
    recording_intro = False

    for line in lines:
        # Start recording Abstract
        if "abstract" in line.lower():
            recording_abstract = True
            continue

        # Start recording Introduction
        if "introduction" in line.lower() and not recording_intro:
            recording_abstract = False
            recording_intro = True
            continue

        # Stop recording if a new section starts
        if recording_intro and re.match(r"^\d+\.\s+[A-Z]", line):
            break

        # Record lines
        if recording_abstract or recording_intro:
            context_lines.append(line.strip())

    return " ".join(context_lines).strip()


def extract_text_from_pdf(pdf_path):
    """
    Extract text, title, context, and body from a PDF file using PyMuPDF.
    """
    try:
        with fitz.open(pdf_path) as pdf:
            first_page = pdf[0]
            text = first_page.get_text("text")

            # Handle cases where the first page has no text
            if not text.strip():
                print(f"Warning: No text found on the first page of {pdf_path}")
                return None

            lines = text.split("\n")

            # Extract context (Abstract + Introduction)
            context = extract_context(lines)

            # Extract full text
            full_text = ""
            for page in pdf:
                full_text += page.get_text("text") + "\n"

            title = os.path.basename(pdf_path).replace("_", " ").replace(".pdf", "")

            return {
                "title": title.strip(),
                "context": context or "",
                "body": full_text.strip(),
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
