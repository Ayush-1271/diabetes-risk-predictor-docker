# Lightweight Docker image for the Diabetes progression prediction Flask API
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY diabetes_model.pkl .
COPY templates/ templates/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PO