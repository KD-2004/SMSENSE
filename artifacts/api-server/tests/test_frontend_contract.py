from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FRONTEND = REPOSITORY_ROOT / "index.html"


def test_frontend_defaults_to_local_api_and_normalizes_trailing_slashes():
    source = FRONTEND.read_text(encoding="utf-8")
    assert 'window.SMSENSE_API_BASE ?? "http://127.0.0.1:4444"' in source
    assert r'.replace(/\/+$/, "")' in source
    assert "ngrok-free" not in source


def test_frontend_does_not_send_ngrok_browser_warning_header():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "ngrok-skip-browser-warning" not in source
