"""Optional Streamlit screen tests; ML contract is tested in test_review.py."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_review import FakeClassifier, picture

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    importlib.util.find_spec("streamlit"), "Optional Streamlit environment"
)
class StreamlitTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st

        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)

    def test_empty_screen_does_not_load_models(self):
        with patch("anomaly_eval.review.Reviewer") as model:
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button[0].disabled)
            self.assertEqual(app.title[0].value, "A closer look.")
            model.assert_not_called()

    def test_uploaded_image_shows_results_and_resets_on_new_file(self):
        from anomaly_eval.review import Reviewer

        engine = Reviewer(FakeClassifier())
        uploaded = SimpleNamespace(name="scene.png", type="image/png", getvalue=picture)
        with (
            patch("streamlit.file_uploader", return_value=uploaded),
            patch("anomaly_eval.review.Reviewer", return_value=engine),
        ):
            app = self.app().run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["result"]["report"]["kind"], "image")
            text = " ".join(m.value for m in app.markdown)
            self.assertIn("More evidence needed", text)
            self.assertIn("No signal", text)
            json.dumps(app.session_state["result"]["report"], allow_nan=False)
            uploaded.name = "different.png"
            app.run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)

    def test_model_errors_do_not_leave_old_success_results(self):
        uploaded = SimpleNamespace(name="scene.png", type="image/png", getvalue=picture)
        with (
            patch("streamlit.file_uploader", return_value=uploaded),
            patch(
                "anomaly_eval.review.Reviewer",
                side_effect=FileNotFoundError("missing weights"),
            ),
        ):
            app = self.app().run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)
            self.assertIn("No result was recorded", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
