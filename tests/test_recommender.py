import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from streamlit.testing.v1 import AppTest

from recommender import Catalog, load_catalog

ROOT = Path(__file__).resolve().parents[1]


class RecommendationTests(unittest.TestCase):
    def test_identical_vectors_do_not_return_the_selected_movie(self):
        catalog = Catalog(
            tuple({"movie_id": i} for i in range(3)),
            sparse.csr_matrix([[1, 0], [1, 0], [0, 1]]),
        )
        self.assertEqual(catalog.recommendations(1), [(0, 1.0), (2, 0.0)])

    def test_missing_tag_vector_has_no_arbitrary_recommendations(self):
        catalog = Catalog(
            tuple({"movie_id": i} for i in range(2)), sparse.csr_matrix([[0, 0], [1, 0]])
        )
        self.assertEqual(catalog.recommendations(0), [])

    def test_invalid_selection_does_not_use_python_negative_indexing(self):
        catalog = Catalog(({}, {}), sparse.csr_matrix(np.eye(2)))
        for invalid in [-1, 2, "Avatar"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                catalog.recommendations(invalid)

    def test_remakes_keep_their_own_cast_and_year(self):
        catalog = load_catalog()
        batman = {movie["movie_id"]: movie for movie in catalog.movies if movie["title"] == "Batman"}
        self.assertEqual(set(batman), {268, 2661})
        self.assertEqual(batman[268]["year"], "1989")
        self.assertEqual(batman[2661]["year"], "1966")
        self.assertIn("Michael Keaton", batman[268]["cast"])
        self.assertIn("Adam West", batman[2661]["cast"])


class InterfaceTests(unittest.TestCase):
    def test_selection_recommendation_and_stale_result_reset(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        self.assertIn("Avatar", app.subheader[0].value)

        catalog = load_catalog()
        second = next(i for i, movie in enumerate(catalog.movies) if movie["movie_id"] == 2661)
        app.selectbox[0].select(second).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        self.assertIn("Batman (1966)", app.subheader[0].value)


if __name__ == "__main__":
    unittest.main()
