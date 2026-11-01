# Stage 1: build the React web app into /public
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python runtime with the Flask API, Streamlit dashboard and MLflow UI
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VOLTCAST_LOG_DIR=/app/logs

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY configs/ configs/
COPY src/ src/
COPY serving/ serving/
COPY app.py .
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/
COPY models/ models/
COPY reports/ reports/
COPY data/processed/hourly.parquet data/processed/hourly.parquet
COPY data/serving/ data/serving/
COPY --from=web /public public/

EXPOSE 8000 8501 5000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
# One worker keeps Prometheus counters in a single process; threads handle concurrent requests.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8000", "app:app"]
