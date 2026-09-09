"""
build_model.py
---------------
Offline pipeline that turns the raw TMDB 5000 CSVs into a single SQLite
database (movies.db) the Flask app can query instantly at runtime.

What it does, and why it's better than the original notebook approach:

1. Cleans and merges the movies + credits datasets.
2. Builds a *weighted* text "soup" per movie instead of a flat bag of
   words — overview text, genres, keywords, top cast and the director
   are each repeated a different number of times so the vectorizer
   treats them with different importance (director/genre matches say
   more about taste than a single overview word).
3. Vectorizes with TF-IDF (unigrams + bigrams) instead of a plain
   CountVectorizer, so common-but-uninformative terms are downweighted
   automatically.
4. Computes cosine similarity, then for every movie keeps only the
   top-K most similar titles (default 30) instead of storing the full
   dense N x N matrix. For ~4800 movies the full matrix is small
   enough to fit in memory here, but shipping only the top-K neighbors
   to the database keeps the web app's memory footprint tiny and lets
   it scale to a much bigger catalog later without changing the app.
5. Computes an IMDB-style Bayesian "weighted rating" (WR) so the app
   can do genuinely hybrid ranking: blend "similar in content" with
   "actually good," instead of just returning the closest match even
   if it's an obscure, poorly-rated film.
6. Stores everything (movie metadata + precomputed neighbor lists) in
   SQLite so the Flask app never has to load pandas/sklearn or hold a
   similarity matrix in memory at request time.

Run once:  python build_model.py
Produces:  movie_rec_system/movies.db
"""

import ast
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "movies.db"

TOP_K_NEIGHBORS = 30       # how many similar movies to precompute per movie
MIN_VOTES_PERCENTILE = 0.60  # 'm' cutoff for the weighted-rating formula


# --------------------------------------------------------------------------
# Helpers to safely parse the TMDB JSON-ish string columns
# --------------------------------------------------------------------------
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_names(value, limit=None):
    items = safe_literal_eval(value)
    names = [d.get("name", "") for d in items if isinstance(d, dict)]
    return names[:limit] if limit else names


def extract_director(crew_value):
    items = safe_literal_eval(crew_value)
    for person in items:
        if isinstance(person, dict) and person.get("job") == "Director":
            return person.get("name", "")
    return ""


def extract_writers(crew_value, limit=2):
    items = safe_literal_eval(crew_value)
    writers = [
        p.get("name", "")
        for p in items
        if isinstance(p, dict) and p.get("department") == "Writing"
    ]
    return writers[:limit]


def clean_token(text):
    # Collapse "Christopher Nolan" -> "christophernolan" so the vectorizer
    # treats it as one token and doesn't confuse "Chris Evans" with
    # "Chris Pratt" just because they share a first name.
    return text.replace(" ", "").lower()


def main():
    print("Loading CSVs...")
    movies = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")

    print("Merging movies + credits...")
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    # Keep only rows with enough signal to be useful
    df = df.dropna(subset=["title", "overview"])
    df["overview"] = df["overview"].fillna("")
    df["vote_count"] = df["vote_count"].fillna(0).astype(int)
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["popularity"] = df["popularity"].fillna(0.0)
    df["runtime"] = df["runtime"].fillna(0).astype(int)
    df["release_date"] = df["release_date"].fillna("")

    print("Parsing genres, keywords, cast, crew...")
    df["genres_list"] = df["genres"].apply(lambda x: extract_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: extract_names(x))
    df["cast_list"] = df["cast"].apply(lambda x: extract_names(x, limit=5))
    df["director"] = df["crew"].apply(extract_director)
    df["writers_list"] = df["crew"].apply(extract_writers)

    print("Building weighted text soup...")

    def build_soup(row):
        genre_tokens = [clean_token(g) for g in row["genres_list"]] * 3
        keyword_tokens = [clean_token(k) for k in row["keywords_list"]] * 2
        cast_tokens = [clean_token(c) for c in row["cast_list"]] * 2
        director_tokens = [clean_token(row["director"])] * 3 if row["director"] else []
        writer_tokens = [clean_token(w) for w in row["writers_list"]]
        overview_tokens = row["overview"].lower().split()

        return " ".join(
            genre_tokens
            + keyword_tokens
            + cast_tokens
            + director_tokens
            + writer_tokens
            + overview_tokens
        )

    df["soup"] = df.apply(build_soup, axis=1)

    # Deduplicate titles that map to the same TMDB id after the merge
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"Vectorizing {len(df)} movies with TF-IDF (uni+bigrams)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=25000,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(df["soup"])

    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    print(f"Reducing to top-{TOP_K_NEIGHBORS} neighbors per movie...")
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, 1 : TOP_K_NEIGHBORS + 1]
    top_k_scores = np.take_along_axis(sim_matrix, top_k_indices, axis=1)

    print("Computing IMDB-style weighted ratings (hybrid signal)...")
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(MIN_VOTES_PERCENTILE)

    def weighted_rating(row, m=m, C=C):
        v = row["vote_count"]
        R = row["vote_average"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    df["weighted_rating"] = df.apply(weighted_rating, axis=1)

    print(f"Writing database to {DB_PATH} ...")
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE movies (
            row_id INTEGER PRIMARY KEY,   -- position in the similarity matrix
            movie_id INTEGER UNIQUE,      -- TMDB id
            title TEXT,
            overview TEXT,
            genres TEXT,                  -- JSON list
            keywords TEXT,                -- JSON list
            cast TEXT,                    -- JSON list
            director TEXT,
            release_date TEXT,
            runtime INTEGER,
            vote_average REAL,
            vote_count INTEGER,
            popularity REAL,
            weighted_rating REAL,
            poster_path TEXT              -- filled in lazily by the app, cached
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE recommendations (
            row_id INTEGER,               -- source movie's row_id
            rank INTEGER,
            rec_row_id INTEGER,           -- recommended movie's row_id
            score REAL,
            PRIMARY KEY (row_id, rank)
        )
        """
    )

    for row_id, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO movies
            (row_id, movie_id, title, overview, genres, keywords, cast,
             director, release_date, runtime, vote_average, vote_count,
             popularity, weighted_rating, poster_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                int(row_id),
                int(row["id"]),
                row["title"],
                row["overview"],
                json.dumps(row["genres_list"]),
                json.dumps(row["keywords_list"]),
                json.dumps(row["cast_list"]),
                row["director"],
                row["release_date"],
                int(row["runtime"]),
                float(row["vote_average"]),
                int(row["vote_count"]),
                float(row["popularity"]),
                float(row["weighted_rating"]),
            ),
        )

    for row_id in range(len(df)):
        for rank, (neighbor_idx, score) in enumerate(
            zip(top_k_indices[row_id], top_k_scores[row_id])
        ):
            cur.execute(
                "INSERT INTO recommendations (row_id, rank, rec_row_id, score) "
                "VALUES (?, ?, ?, ?)",
                (row_id, rank, int(neighbor_idx), float(score)),
            )

    cur.execute("CREATE INDEX idx_movies_title ON movies(title COLLATE NOCASE)")
    cur.execute("CREATE INDEX idx_reco_row_id ON recommendations(row_id)")

    conn.commit()
    conn.close()
    print("Done. movies.db is ready for the Flask app.")


if __name__ == "__main__":
    main()
