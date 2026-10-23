from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FRONTEND = REPOSITORY_ROOT / "index.html"


def test_frontend_defaults_to_local_api_but_explains_github_pages_configuration():
    source = FRONTEND.read_text(encoding="utf-8")
    assert 'window.SMSENSE_API_BASE ?? ""' in source
    assert '"http://127.0.0.1:4444"' in source
    assert "isHostedGitHubPages" in source
    assert "API_CONFIG_ERROR" in source
    assert "cannot reach Ollama on your computer through localhost" in source
    assert r'.replace(/\\/+$/, "")' in source
    assert "ngrok-free" not in source


def test_frontend_displays_sms_shield_primary_assessment():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "Primary Model Assessment" in source
    assert "modelAssessment.red_flags" in source
    assert "SMS-Shield" in source


def test_frontend_shows_core_result_before_optional_explanation():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "loadPostHocExplanation(result);" in source
    assert "apiRequest('/api/explain'" in source
    assert "currentAnalysis !== result" in source
    assert "Generating the optional explanation separately" in source


def test_frontend_does_not_send_ngrok_browser_warning_header():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "ngrok-skip-browser-warning" not in source
