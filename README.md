# Movie Name Recommender

A **content-based movie recommendation** application built with Python and Streamlit. Select a movie to discover five films with similar metadata, using **CountVectorizer** and **cosine similarity**.

## How it works

1. Join movie metadata and credits by movie ID, then remove rows with missing required metadata.
2. Build tags from each movie's overview, genres, keywords, first three cast members, and directors.
3. Remove spaces within metadata names and combine the tags into text.
4. Convert the tags into word-count vectors with `CountVectorizer(max_features=5000, stop_words="english")`.
5. Compare the selected movie's vector with the catalogue using cosine similarity and return the five closest movies, excluding the selected title's own record.

Vectors are stored as a sparse matrix. Each request computes one row of similarity scores instead of loading a full pairwise similarity matrix.

## Features

- Searchable movie selector with release years to distinguish shared titles.
- Five recommendations with titles, genres, overviews, cast, and tag-similarity scores.
- Prepared data included in the repository; no API key is required to run the app.
- Asset integrity checks for catalogue and vector alignment.
- Separate recommendation logic and Streamlit interface.

**Tag similarity** measures overlap in movie content. It is not a rating or a probability that someone will enjoy a film. Recommendations use movie metadata rather than watch history or other users' ratings.

## Run locally

Use **Python 3.11**. Application dependencies are pinned in `requirements.txt`.

### Windows PowerShell

```powershell
git clone https://github.com/Siddharth-smnt/movie-recommender.git
cd movie-recommender
py -3.11 -m venv .venv-movies
.\.venv-movies\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-movies\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/Siddharth-smnt/movie-recommender.git
cd movie-recommender
python3.11 -m venv .venv-movies
.venv-movies/bin/python -m pip install -r requirements.txt
.venv-movies/bin/python -m streamlit run app.py
```

Open the address printed by Streamlit, normally [http://localhost:8501](http://localhost:8501). Choose a film and select **Find recommendations**.

## Dataset and prepared assets

The project uses the [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), based on metadata from The Movie Database.

The included build contains **4,800 movies** and a **4,800 × 5,000** feature matrix, as recorded in `data/build_info.json`. These are catalogue dimensions, not evaluation scores.

The `data/` directory contains the movie catalogue, sparse vectors, vocabulary, and build metadata. The application reads these local files and does not call Kaggle or TMDB at runtime. The catalogue is a fixed dataset and does not include every film or recent release.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit selector and recommendation cards |
| `recommender.py` | Catalogue validation and cosine-similarity ranking |
| `prepare_data.py` | Build tags and sparse vectors from the source CSVs |
| `data/movies.json` | Movie IDs, titles, years, and display metadata |
| `data/vectors.npz` | Compressed sparse word-count vectors |
| `data/vocabulary.json` | Feature vocabulary |
| `data/build_info.json` | Build parameters, dimensions, and artifact hashes |
| `notebooks/original.ipynb` | Original exploratory notebook |
| `verify_project.py` | Data and implementation checks |
| `tests/test_recommender.py` | Ranking and interface regression tests |
| `validation/verification.json` | Recorded verification results |

## Rebuild the data

The prepared assets are sufficient to run the app. To rebuild them, download the Kaggle dataset and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside `raw_data/`.

In Windows PowerShell, run:

```powershell
.\.venv-movies\Scripts\python.exe prepare_data.py
.\.venv-movies\Scripts\python.exe verify_project.py --movies raw_data/tmdb_5000_movies.csv --credits raw_data/tmdb_5000_credits.csv
```

On macOS/Linux, use `.venv-movies/bin/python`. Rebuilding replaces the files in `data/`; keep the generated files together so their recorded hashes and row ordering remain consistent.

## Checks and implementation notes

From the project directory in Windows PowerShell:

```powershell
.\.venv-movies\Scripts\python.exe -m pip check
.\.venv-movies\Scripts\python.exe verify_project.py
.\.venv-movies\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS/Linux, use `.venv-movies/bin/python` for these commands.

The web implementation preserves the notebook's tag construction and count-vector approach. It joins credits by movie ID to keep different films with the same title separate, resets row positions after filtering, and explicitly excludes the selected record. These corrections can change recommendations compared with the original notebook's title-based join.

The checks cover implementation consistency and application behavior. Recommendation quality has not been measured with a user study or a held-out relevance benchmark.

## Technology and attribution

Python · Streamlit · pandas · NumPy · SciPy · scikit-learn

Movie metadata is sourced from TMDB via Kaggle. This project is not endorsed or certified by TMDB.

Developed by [Siddharth Samant](https://github.com/Siddharth-smnt) · [Portfolio](https://www.whoissid.com)
