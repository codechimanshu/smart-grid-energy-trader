# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Keep the container running in the background for the hackathon evaluator
CMD ["python", "-m", "http.server", "7860"]