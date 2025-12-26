# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
# We need these for Playwright and general build tools
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (required for Crawl4AI)
RUN playwright install --with-deps chromium

# Download NLTK data (punkt, stopwords)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Expose port (RunPod/Railway usually map internal 8080/Configured)
EXPOSE 8080

# Command to run the application using Gunicorn
# Adjust workers based on core count (2-4 usually good)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.web_app:app
