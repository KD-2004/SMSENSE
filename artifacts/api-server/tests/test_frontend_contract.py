from pathlib import Path
import re
import shutil
import subprocess
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FRONTEND = REPOSITORY_ROOT / "index.html"


def test_frontend_defaults_to_local_api_but_explains_github_pages_configuration():
    source = FRONTEND.read_text(encoding="utf-8")
    assert 'window.SMSENSE_API_BASE ?? ""' in source
    assert '"http://127.0.0.1:4444"' in source
    assert "isHostedGitHubPages" in source
    assert "API_CONFIG_ERROR" in source
    assert "cannot reach Ollama on your computer through localhost" in source
    assert r'.replace(/\/+$/, "")' in source
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


def test_inline_frontend_javascript_has_valid_syntax():
    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("Node.js is unavailable; frontend syntax check skipped")

    source = FRONTEND.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", source, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "Expected an inline frontend script"

    with tempfile.TemporaryDirectory() as temp_dir:
        for index, script in enumerate(scripts):
            script_path = Path(temp_dir) / f"frontend-{index}.js"
            script_path.write_text(script, encoding="utf-8")
            checked = subprocess.run(
                [node, "--check", str(script_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert checked.returncode == 0, checked.stderr


def test_frontend_explains_similarity_is_diagnostic_only_when_disabled():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "result.similarity?.scoring_enabled === false" in source
    assert "excluded from this risk score until it is validated" in source


def test_frontend_does_not_send_ngrok_browser_warning_header():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "ngrok-skip-browser-warning" not in source



def test_frontend_prevents_overlapping_analysis_and_resets_cleared_selection():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "if (btn.disabled) return;" in source
    assert 'id="clearAnalyzeBtn"' in source
    assert "if (clearBtn) clearBtn.disabled = true;" in source
    clear_fn = re.search(
        r"function clearAnalyzeForm\(\)\s*\{(.*?)\n\s*\}",
        source,
        flags=re.DOTALL,
    )
    assert clear_fn, "Expected a clearAnalyzeForm implementation"
    assert "currentAnalysis = null;" in clear_fn.group(1)
    assert "currentAnalysisId = null;" in clear_fn.group(1)
    assert "renderModels();" in clear_fn.group(1)


def test_frontend_feedback_requires_a_saved_analysis_id():
    source = FRONTEND.read_text(encoding="utf-8")
    assert 'id="submitFeedbackBtn"' in source
    assert "Number.isInteger(Number(currentAnalysisId))" in source
    assert "Feedback requires a saved analysis ID" in source
    assert "analysis_id: Number(currentAnalysisId)" in source



def test_frontend_renders_a_realistic_backend_result_instead_of_only_parsing_it():
    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("Node.js is unavailable; frontend rendering smoke test skipped")

    source = FRONTEND.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", source, flags=re.DOTALL | re.IGNORECASE)
    frontend_script = next((part for part in scripts if "function renderAnalysisResult" in part), None)
    assert frontend_script, "Expected the frontend result-rendering function"

    harness = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const script = fs.readFileSync(process.argv[2], "utf8");
const fakeDocument = {
  addEventListener() {},
  createElement() {
    let safeHtml = "";
    return {
      set textContent(value) {
        safeHtml = String(value).replace(/[&<>"']/g, ch => ({
          "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        })[ch]);
      },
      get innerHTML() { return safeHtml; }
    };
  }
};
const sandbox = {
  window: { SMSENSE_API_BASE: "", location: { hostname: "localhost" } },
  document: fakeDocument,
  console,
  Headers,
  AbortController,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  fetch: async () => { throw new Error("fetch must not run during rendering smoke test"); }
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox, { timeout: 5000 });
const container = { innerHTML: "" };
sandbox.renderAnalysisResult({
  id: 123,
  risk_score: 87,
  label: "CRITICAL",
  category: "KYC_FRAUD",
  language: "Hinglish",
  raw_message: "Your account will be blocked; send your PIN.",
  normalized_message: "your account will be blocked; send your pin.",
  sender: "VM-SBIHELP",
  model_assessment: {
    model: "SMS-Shield", risk: 91, verdict: "dangerous",
    explanation: "Requests sensitive credentials.",
    red_flags: ["PIN request"], segments_analyzed: 1
  },
  decision_diagnostics: {},
  evidence: { content: 27, sender: 4, url: 8, behavior: 18, obfuscation: 6, similarity: 0 },
  suspicious_phrases: ["KYC plus account threat"],
  url_analysis: { count: 0, urls: [] },
  obfuscation: {},
  explanation: { top_contributing_evidence: ["content"], model_signals: [
    { name: "SMS-Shield", available: true, score: 13.65, detail: "risk=91/100" }
  ] },
  post_hoc: null
}, container);
for (const expected of [
  "Analysis Result", "Primary Model Assessment — SMS-Shield",
  "Model risk score: 91 / 100", "KYC_FRAUD",
  "Your account will be blocked; send your PIN.", "KYC plus account threat"
]) {
  if (!container.innerHTML.includes(expected)) {
    throw new Error("Rendered analysis is missing expected content: " + expected);
  }
}
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "frontend.js"
        harness_path = Path(temp_dir) / "render-smoke.js"
        script_path.write_text(frontend_script, encoding="utf-8")
        harness_path.write_text(harness, encoding="utf-8")
        checked = subprocess.run(
            [node, str(harness_path), str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr



def test_history_detail_requests_missing_optional_explanation():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "function loadHistoryDetail(id)" in source
    assert "if (!analysis.post_hoc?.summary) loadPostHocExplanation(analysis);" in source
