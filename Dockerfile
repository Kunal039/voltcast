FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY configs/ configs/
COPY src/ src/
COPY serving/ serving/
COPY public/ public/
COPY app.py .
COPY dashboard/ dashboard/
COPY models/ models/
COPY reports/ reports/
COPY data/processed/hourly.parquet data/processed/hourly.parquet
COPY data/serving/ data/serving/

EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
