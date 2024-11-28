# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app/api/

# Copy the requirements file into the container
COPY ./api/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API folder
COPY api /app/api/

# Copy the embeddings folder
COPY embeddings /app/embeddings/

# Copy the collected-data folder
COPY collected-data /app/collected-data/

# Expose the port your backend API runs on
EXPOSE 8000

# Command to run the backend with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
