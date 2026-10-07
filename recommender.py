"""Load the prepared movie tags and rank them using the notebook's cosine metric."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Catalog:
    movies: tuple[dict, ...]
    vectors: sparse.csr_matrix

    def label(self, index: int) -> str:
        movie = self.movies[index]
        return f"{movie['title']} ({movie['year'] or 'year unknown'})"

    def recommendations(self, index: int, count: int = 5) -> list[tuple[int, float]]:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.movies):
            raise ValueError("Choose a movie from the catalogue.")
        if not isinstance(count, int) or count < 1:
            raise ValueError("The recommendation count must be a positive integer.")
        if self.vectors[index].nnz == 0:
            return []

        # Compute one row on demand instead of storing a dense N × N matrix.
        scores = cosine_similarity(self.vectors[index:index + 1], self.vectors).ravel()
        ranked = sorted(
            ((position, float(score)) for position, score in enumerate(scores) if position != index),
            key=lambda pair: pair[1], reverse=True,
        )
        # Python's stable sort keeps original row order for exact ties, as in the notebook.
        return ranked[:count]


def load_catalog(data_dir: Path = DATA_DIR) -> Catalog:
    data_dir = Path(data_dir)
    info = json.loads((data_dir / "build_info.json").read_text(encoding="utf-8"))
    if info.get("schema_version") != 1:
        raise ValueError("Unsupported data format; rebuild the deployment assets.")
    for name in ["movies.json", "vectors.npz"]:
        with (data_dir / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != info["artifacts"][name]["sha256"]:
            raise ValueError(f"{name} differs from its build manifest; restore or rebuild all data files together.")

    movies = json.loads((data_dir / "movies.json").read_text(encoding="utf-8"))
    vectors = sparse.csr_matrix(sparse.load_npz(data_dir / "vectors.npz"))
    if not isinstance(movies, list) or len(movies) < 2:
        raise ValueError("The movie catalogue must contain at least two records.")
    if vectors.shape != tuple(info["feature_shape"]) or len(movies) != vectors.shape[0]:
        raise ValueError("Movie records and feature vectors are not aligned.")
    required = {"movie_id", "title", "year", "overview", "genres", "cast", "directors"}
    if any(not isinstance(movie, dict) or not required.issubset(movie) for movie in movies):
        raise ValueError("Movie metadata is incomplete.")
    if len({movie["movie_id"] for movie in movies}) != len(movies):
        raise ValueError("Movie IDs must be unique.")
    if not np.isfinite(vectors.data).all() or (vectors.data < 0).any():
        raise ValueError("Feature vectors must contain finite, nonnegative counts.")
    return Catalog(tuple(movies), vectors)
