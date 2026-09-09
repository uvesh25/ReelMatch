"""
app.py
------
Flask backend for the movie recommendation system.

Design choices vs. the original Streamlit + pickle version:

- No pandas/sklearn/pickle loaded at request time. build_model.py already
  did that work offline; this process only ever talks to SQLite, so it
  starts instantly and uses very little memory.
- Recommendations are re-ranked with a hybrid score: 70% content
  similarity + 30% the movie's IMDB-style weighted rating (normalized).
  This keeps recommendations relevant AND filters out the case where the
  closest "similar" movie is some obscure, badly-rated title that just
  happens to share a keyword.
- Poster URLs are fetched from TMDB once per movie and cached in the
  database, so repeat requests never re-hit the external API.
- A real REST API (JSON) backs a plain HTML/CSS/JS frontend, which is the
  full-stack shape (frontend + backend + DB) rather than a single
  Streamlit script.
"""

import os
import sqlite3
from pathlib import Path

import requests
from flask import Flask, g, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "movies.db"

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/342x513/141018/f5c451?text=No+Poster"

app = Flask(__name__)


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def row_to_movie(row, include_overview=False):
    import json

    movie = {
        "movie_id": row["movie_id"],
        "row_id": row["row_id"],
        "title": row["title"],
        "genres": json.loads(row["genres"]),
        "director": row["director"],
        "release_date": row["release_date"],
        "release_year": (row["release_date"] or "")[:4],
        "runtime": row["runtime"],
        "vote_average": round(row["vote_average"], 1),
        "vote_count": row["vote_count"],
        "poster_url": get_poster_url(row["movie_id"], row["poster_path"]),
    }
    if include_overview:
        movie["overview"] = row["overview"]
        movie["cast"] = json.loads(row["cast"])
        movie["keywords"] = json.loads(row["keywords"])
    return movie


# --------------------------------------------------------------------------
# Poster fetching with DB-backed caching
# --------------------------------------------------------------------------
def get_poster_url(movie_id, cached_poster_path):
    if cached_poster_path == "__NONE__":
        return PLACEHOLDER_POSTER
    if cached_poster_path:
        return f"{TMDB_IMAGE_BASE}{cached_poster_path}"

    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER

    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=4,
        )
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
    except (requests.RequestException, ValueError):
        poster_path = None

    db = get_db()
    db.execute(
        "UPDATE movies SET poster_path = ? WHERE movie_id = ?",
        (poster_path if poster_path else "__NONE__", movie_id),
    )
    db.commit()

    return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else PLACEHOLDER_POSTER


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trending")
def trending():
    """Homepage rail: highest weighted-rating movies, i.e. genuinely good,
    not just popular/loud."""
    limit = int(request.args.get("limit", 18))
    db = get_db()
    rows = db.execute(
        "SELECT * FROM movies ORDER BY weighted_rating DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return jsonify([row_to_movie(r) for r in rows])


@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    db = get_db()
    rows = db.execute(
        "SELECT * FROM movies WHERE title LIKE ? "
        "ORDER BY popularity DESC LIMIT 8",
        (f"%{q}%",),
    ).fetchall()
    return jsonify([row_to_movie(r) for r in rows])


@app.route("/api/movie/<int:movie_id>")
def movie_detail(movie_id):
    db = get_db()
    row = db.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(row_to_movie(row, include_overview=True))


@app.route("/api/recommend/<int:movie_id>")
def recommend(movie_id):
    limit = int(request.args.get("limit", 10))
    db = get_db()

    source = db.execute(
        "SELECT row_id FROM movies WHERE movie_id = ?", (movie_id,)
    ).fetchone()
    if source is None:
        return jsonify({"error": "Movie not found"}), 404

    row_id = source["row_id"]

    # Pull more neighbors than needed so the hybrid re-rank has room to
    # promote a slightly-less-similar-but-much-better-rated movie.
    candidates = db.execute(
        """
        SELECT r.score, m.*
        FROM recommendations r
        JOIN movies m ON m.row_id = r.rec_row_id
        WHERE r.row_id = ?
        ORDER BY r.rank
        LIMIT 25
        """,
        (row_id,),
    ).fetchall()

    if not candidates:
        return jsonify([])

    max_rating = max(c["weighted_rating"] for c in candidates) or 1.0
    max_sim = max(c["score"] for c in candidates) or 1.0

    scored = []
    for c in candidates:
        sim_norm = c["score"] / max_sim
        rating_norm = c["weighted_rating"] / max_rating
        hybrid = 0.7 * sim_norm + 0.3 * rating_norm
        scored.append((hybrid, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]

    results = []
    for hybrid_score, row in top:
        movie = row_to_movie(row)
        movie["match_score"] = round(hybrid_score * 100)
        results.append(movie)

    return jsonify(results)


@app.route("/api/genres")
def genres():
    """All distinct genres, for the browse-by-genre filter."""
    import json

    db = get_db()
    rows = db.execute("SELECT genres FROM movies").fetchall()
    seen = set()
    for r in rows:
        for genre in json.loads(r["genres"]):
            seen.add(genre)
    return jsonify(sorted(seen))


@app.route("/api/genre/<genre_name>")
def by_genre(genre_name):
    limit = int(request.args.get("limit", 18))
    db = get_db()
    rows = db.execute(
        "SELECT * FROM movies WHERE genres LIKE ? "
        "ORDER BY weighted_rating DESC LIMIT ?",
        (f'%"{genre_name}"%', limit),
    ).fetchall()
    return jsonify([row_to_movie(r) for r in rows])


if __name__ == "__main__":
    if not DB_PATH.exists():
        raise SystemExit(
            "movies.db not found. Run `python build_model.py` first."
        )
    app.run(debug=True, port=5000)
