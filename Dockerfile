# Dockerfile
FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD echo "Building Docker image for Run ID: ${RUN_ID}" && echo "Mock deployment complete"