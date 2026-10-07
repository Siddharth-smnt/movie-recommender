"""Streamlit interface for the notebook's content-based movie recommender."""

import logging

import streamlit as st

from recommender import load_catalog

st.set_page_config(page_title="Movie Name Recommender", page_icon="🍿", layout="centered")


@st.cache_resource(show_spinner=False)
def get_catalog():
    return load_catalog()


def clear_recommendations():
    st.session_state.pop("recommended_index", None)


st.caption("SID'S PROJECTS / MOVIE DISCOVERY")
st.title("Find your next movie 🍿")
st.write("Pick a film you enjoyed. Discover five movies with similar stories, genres, and cast.")

try:
    with st.spinner("Loading the movie catalogue…"):
        catalog = get_catalog()
except Exception:
    logging.exception("Unable to load recommendation assets")
    st.error("The movie catalogue is unavailable. Please refresh or try again later.")
    st.stop()

options = sorted(range(len(catalog.movies)), key=lambda i: (catalog.label(i).casefold(), i))
default_row = next((i for i, movie in enumerate(catalog.movies) if movie["movie_id"] == 19995), options[0])
selected = st.selectbox(
    "Choose a movie", options=options, index=options.index(default_row),
    format_func=catalog.label, key="selected_movie", on_change=clear_recommendations,
    help="Start typing a title to search. Release years distinguish films with the same name.",
)
st.caption(f"Search {len(catalog.movies):,} movies from the TMDB dataset.")

if st.button("Find recommendations", type="primary", width="stretch"):
    st.session_state["recommended_index"] = selected

with st.expander("About your selection"):
    chosen = catalog.movies[selected]
    st.write(chosen["overview"])
    if chosen["genres"]:
        st.caption("Genres: " + " · ".join(chosen["genres"]))
    if chosen["cast"]:
        st.caption("Cast: " + ", ".join(chosen["cast"]))
    if chosen["directors"]:
        st.caption("Director: " + ", ".join(chosen["directors"]))

query = st.session_state.get("recommended_index")
if query is not None:
    matches = catalog.recommendations(query)
    st.divider()
    st.subheader(f"Because you chose {catalog.label(query)}")
    if not matches:
        st.info("There isn't enough tag information for this movie. Please choose another title.")
    else:
        st.caption("Tag similarity compares movie content on a 0–1 scale. It is not a rating or a probability that you will like a film.")
        for rank, (index, score) in enumerate(matches, start=1):
            movie = catalog.movies[index]
            with st.container(border=True):
                title_column, score_column = st.columns([4, 1])
                with title_column:
                    st.subheader(f"{rank}. {movie['title']}")
                    st.caption(" · ".join([movie["year"] or "Year unknown", *movie["genres"]]))
                with score_column:
                    st.metric("Tag similarity", f"{score:.3f}")
                st.write(movie["overview"])
                if movie["cast"]:
                    st.caption("Starring " + ", ".join(movie["cast"]))
else:
    st.info("Choose a movie, then select Find recommendations.")

st.divider()
with st.expander("How the recommendations work"):
    st.write(
        "Each movie is represented by tags from its overview, genres, keywords, "
        "first three cast members, and directors. The tags become word-count vectors "
        "with a vocabulary of up to 5,000 terms. Cosine similarity finds the five "
        "closest movies, excluding the selected movie itself."
    )
    st.write(
        "This is a content-based recommender. It does not learn from your watch "
        "history or other users' ratings. The catalogue is a fixed dataset, so "
        "some films and recent releases are unavailable."
    )
st.caption("Movie metadata: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).")
st.caption("This demo uses TMDB metadata via Kaggle and is not endorsed or certified by TMDB.")
