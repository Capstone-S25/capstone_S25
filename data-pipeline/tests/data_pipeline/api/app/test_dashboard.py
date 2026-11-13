from fastapi.testclient import TestClient

from data_pipeline.api.app import create_app

# No request here reaches the database; the engine connects lazily.
URL = "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"


def test_built_dashboard_is_served_with_client_side_routes(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>dashboard</html>")
    (dist / "assets/app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside")
    monkeypatch.setenv("DASHBOARD_DIR", str(dist))
    with TestClient(create_app(URL)) as client:
        assert "dashboard" in client.get("/").text
        # React routes are not files; they get the app shell.
        assert "dashboard" in client.get("/staff/home").text
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/api/v1/unknown").status_code == 404
        assert "outside" not in client.get("/../secret.txt").text
        assert client.get("/openapi.json").status_code == 200


def test_without_a_dashboard_folder_only_the_api_exists(monkeypatch):
    monkeypatch.delenv("DASHBOARD_DIR", raising=False)
    with TestClient(create_app(URL)) as client:
        assert client.get("/staff/home").status_code == 404
