FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-download models
RUN python3 -c "from transformers import BertTokenizer, BertModel; \
    BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False); \
    BertModel.from_pretrained('bert-base-uncased', local_files_only=False); \
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('bert-base-nli-mean-tokens', local_files_only=False); \
    SentenceTransformer('all-MiniLM-L6-v2', local_files_only=False)"

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Expose the port
EXPOSE 5000

# Command to run the application with optimized settings
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "1", "--timeout", "120", "--max-requests", "100", "--max-requests-jitter", "10", "--worker-class", "sync", "--log-level", "info", "app:app"]
