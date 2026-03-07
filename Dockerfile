FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy base model (used as fallback if fine-tuned models are absent)
RUN python -m spacy download fr_core_news_sm

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
