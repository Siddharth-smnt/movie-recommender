"""Verify the packaged assets; optionally compare against the supplied notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from prepare_data import prepare_table
from recommender import load_catalog

ROOT = Path(__file__).resolve().parent


def compare_with_notebook(movies_path: Path, credits_path: Path, catalog) -> dict:
    """Run the reviewed transformation cells, changing only the broken title join."""
    notebook = json.loads((ROOT / "notebooks" / "original.ipynb").read_text(encoding="utf-8"))
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    joined = movies.merge(
        credits.drop(columns="title"), left_on="id", right_on="movie_id",
        validate="one_to_one", sort=False,
    )
    scope = {"movies": joined, "np": np, "pd": pd}
    # These are the exact tag-construction and vectorizer-definition cells supplied
    # by the user. Loading, plotting, the title join, and pickle export are excluded.
    cells = [7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 28]
    for number in cells:
        source = "".join(notebook["cells"][number]["source"])
        exec(compile(source, f"original.ipynb:cell-{number}", "exec"), scope)
    expected = scope["new"].reset_index(drop=True)
    actual, _ = prepare_table(movies_path, credits_path)
    for column in ["movie_id", "title", "tags"]:
        if not actual[column].equals(expected[column]):
            raise AssertionError(f"Notebook comparison failed for {column}.")
    expected_vectors = scope["cv"].fit_transform(expected["tags"])
    if expected_vectors.shape != catalog.vectors.shape or (expected_vectors != catalog.vectors).nnz:
        raise AssertionError("Packaged vectors differ from notebook processing after the ID-join fix.")
    if actual.drop(columns="tags").to_dict("records") != list(catalog.movies):
        raise AssertionError("Packaged movie metadata differs from the CSV build.")
    return {"tag_rows_equal": len(expected), "count_vectors_equal": True, "join_correction_applied": True}


def verify(movies_path: Path | None = None, credits_path: Path | None = None) -> dict:
    catalog = load_catalog()
    # A dispersed sample, including the last rows, checks sparse cosine against the
    # notebook's dense representation without allocating a full N × N matrix.
    positions = np.unique(np.linspace(0, len(catalog.movies) - 1, min(128, len(catalog.movies)), dtype=int))
    sampled = catalog.vectors[positions]
    dense_scores = cosine_similarity(sampled.toarray())
    sparse_scores = cosine_similarity(sampled)
    np.testing.assert_allclose(sparse_scores, dense_scores, rtol=1e-12, atol=1e-12)
    examples = []
    wanted_ids = [19995, 268, 2661, catalog.movies[-1]["movie_id"]]
    for movie_id in wanted_ids:
        index = next(i for i, movie in enumerate(catalog.movies) if movie["movie_id"] == movie_id)
        matches = catalog.recommendations(index)
        if len(matches) != min(5, len(catalog.movies) - 1) or any(i == index for i, _ in matches):
            raise AssertionError("A recommendation list has the wrong size or contains its query.")
        examples.append({
            "query": catalog.label(index),
            "recommendations": [{"movie": catalog.label(i), "cosine_similarity": score} for i, score in matches],
        })
    report = {
        "asset_integrity": "passed",
        "catalog_movies": len(catalog.movies),
        "feature_shape": list(catalog.vectors.shape),
        "sparse_vs_dense_sample_rows": len(positions),
        "max_absolute_cosine_difference": float(np.max(np.abs(sparse_scores - dense_scores))),
        "examples": examples,
    }
    if movies_path is not None and credits_path is not None:
        report["notebook_comparison"] = compare_with_notebook(movies_path, credits_path, catalog)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path)
    parser.add_argument("--credits", type=Path)
    parser.add_argument("--output", type=Path, help="Optional location for the verification JSON.")
    args = parser.parse_args()
    if bool(args.movies) != bool(args.credits):
        parser.error("Supply both --movies and --credits for the notebook comparison.")
    report = verify(args.movies, args.credits)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
