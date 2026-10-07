"""Build small deployment assets from the two CSVs used in the original notebook."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_items(value: str, column: str) -> list[dict]:
    """Use the notebook's literal parsing, with a useful error for malformed data."""
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid list in the {column!r} column.") from exc
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in items
    ):
        raise ValueError(f"Expected a list of named objects in {column!r}.")
    return items


def prepare_table(movies_path: Path, credits_path: Path) -> tuple[pd.DataFrame, dict]:
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    requirements = (
        (movies, {"id", "title", "overview", "genres", "keywords", "release_date"}, "movies"),
        (credits, {"movie_id", "title", "cast", "crew"}, "credits"),
    )
    for frame, required, label in requirements:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"The {label} CSV is missing columns: {', '.join(missing)}")
    if movies["id"].isna().any() or credits["movie_id"].isna().any():
        raise ValueError("Movie IDs must not be missing.")

    # Intentional bug fix: title-based merging confuses remakes with the same title.
    merged = movies.merge(
        credits.rename(columns={"title": "credits_title"}),
        left_on="id", right_on="movie_id", how="inner", sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(movies) or len(merged) != len(credits):
        raise ValueError("Some movie IDs have no matching row in the other CSV.")
    if not merged["title"].equals(merged["credits_title"]):
        raise ValueError("Titles disagree between the movie and credit rows for an ID.")

    selected = ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    missing_mask = merged[selected].isna().any(axis=1)
    dropped = merged.loc[missing_mask, ["movie_id", "title"]].to_dict("records")
    clean = merged.loc[~missing_mask].copy().reset_index(drop=True)
    if len(clean) < 2:
        raise ValueError("At least two movies with complete metadata are required.")

    records = []
    for row in clean.itertuples(index=False):
        genres = [item["name"] for item in parse_items(row.genres, "genres")]
        keywords = [item["name"] for item in parse_items(row.keywords, "keywords")]
        cast = [item["name"] for item in parse_items(row.cast, "cast")][:3]
        crew = parse_items(row.crew, "crew")
        directors = [item["name"] for item in crew if item.get("job") == "Director"]

        # Same order and space removal as cells 21–27 of the supplied notebook.
        tokens = row.overview.split()
        tokens += [name.replace(" ", "") for name in genres + keywords + cast + directors]
        date = row.release_date if isinstance(row.release_date, str) else ""
        year = date[:4] if len(date) >= 4 and date[:4].isdigit() else None
        records.append({
            "movie_id": int(row.movie_id), "title": row.title, "year": year,
            "overview": row.overview, "genres": genres, "cast": cast,
            "directors": directors, "tags": " ".join(tokens),
        })

    report = {
        "source_rows": {"movies": len(movies), "credits": len(credits)},
        "join": "movies.id = credits.movie_id (one-to-one)",
        "joined_rows": len(merged),
        "dropped_missing_metadata": dropped,
        "catalog_rows": len(records),
    }
    return pd.DataFrame(records), report


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_assets(movies_path: Path, credits_path: Path, output: Path) -> dict:
    table, report = prepare_table(movies_path, credits_path)
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = sparse.csr_matrix(vectorizer.fit_transform(table["tags"]))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "movies.json", table.drop(columns="tags").to_dict("records"))
    sparse.save_npz(output / "vectors.npz", vectors, compressed=True)
    write_json(output / "vocabulary.json", vectorizer.get_feature_names_out().tolist())
    report.update({
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_sha256": {
            movies_path.name: sha256(movies_path), credits_path.name: sha256(credits_path),
        },
        "vectorizer": {"class": "CountVectorizer", "max_features": 5000, "stop_words": "english"},
        "feature_shape": list(vectors.shape),
        "feature_dtype": str(vectors.dtype),
        "nonzero_counts": int(vectors.nnz),
        "zero_vector_rows": int((vectors.getnnz(axis=1) == 0).sum()),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "streamlit"]
        },
        "artifacts": {
            name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in ["movies.json", "vectors.npz", "vocabulary.json"]
        },
    })
    notebook = ROOT / "notebooks" / "original.ipynb"
    if notebook.is_file():
        report["original_notebook_sha256"] = sha256(notebook)
    write_json(output / "build_info.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=ROOT / "raw_data" / "tmdb_5000_movies.csv")
    parser.add_argument("--credits", type=Path, default=ROOT / "raw_data" / "tmdb_5000_credits.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    for path in [args.movies, args.credits]:
        if not path.is_file():
            parser.error(f"CSV not found: {path}. Download and extract the dataset linked in README.md.")
    report = build_assets(args.movies, args.credits, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
