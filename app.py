"""Web entrypoint (Vercel / Docker / local): Flask API under /api, React site at /."""
from pathlib import Path

from flask import Flask, send_from_directory

from serving.main import api

PUBLIC = Path(__file__).parent / "public"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # reject oversized request bodies
app.register_blueprint(api, url_prefix="/api")


# On Vercel the CDN serves public/ directly; locally and in Docker Flask serves the built React site.
@app.get("/")
@app.get("/<path:path>")
def web(path: str = "index.html"):
    target = PUBLIC / path
    return send_from_directory(PUBLIC, path if target.is_file() else "index.html")


if __name__ == "__main__":
    app.run(port=8000, debug=False)
