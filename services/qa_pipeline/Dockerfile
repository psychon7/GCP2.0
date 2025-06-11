FROM python:3.11-slim-bookworm

# Update packages and install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m appuser

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the QA pipeline code
COPY . .

# Create directory for config files if it doesn't exist
RUN mkdir -p /app/config && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Use tini as entry point for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Command to run the service
CMD ["python", "main.py", "service"]
