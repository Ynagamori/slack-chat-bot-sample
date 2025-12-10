FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY ingest_docs.py .
COPY query_test.py .

CMD ["python", "main.py"]
