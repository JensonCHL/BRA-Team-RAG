# Dockerfile for Streamlit RAG Application
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependencies first (cached unless changed)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create directories but don't copy application code
# Code will be mounted as a volume in docker-compose
RUN mkdir -p uploads artifacts

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "RAG.py", "--server.port=80", "--server.address=0.0.0.0"]