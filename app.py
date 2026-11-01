"""Web entrypoint (Vercel / Docker): prediction API under /api, web dashboard at /."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving.main import app as api_app

app = FastAPI(title="Household Electricity Consumption Prediction System", docs_url=None, redoc_url=None)
app.mount("/api", api_app)

# On Vercel the CDN serves public/ directly; locally and in Docker the app serves it.
PUBLIC = Path(__file__).parent / "public"
if PUBLIC.exists():
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="web")
