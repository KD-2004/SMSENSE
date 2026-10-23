from app.fusion import fuse
from app.normalizer import normalize_message
from app.parser import parse_message
from app.rules import behavior_signals, _credential_request


def test_parser_preserves_raw_and_extracts_url():
    raw = "  KYC!!! visit https://bit.ly/abc  "
    parsed = parse_message(raw, "BANKALERT", None)
    assert parsed.raw == raw
    assert parsed.domains == ["bit.ly"]
    assert parsed.length == len(raw)


def test_parser_extracts_bare_and_shortened_domains_and_keeps_distinct_paths():
    raw = "Visit bit.ly/abc123, i.airtel.in/RechargeNow and https://example.com/a https://example.com/b"
    parsed = parse_message(raw, None, None)
    assert parsed.urls == [
        "bit.ly/abc123",
        "i.airtel.in/RechargeNow",
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert parsed.domains == ["bit.ly", "i.airtel.in", "example.com"]


def test_url_analysis_checks_each_path_when_domain_is_deduplicated():
    from app.rules import url_signals

    long_url = "https://example.com/" + ("a" * 210)
    parsed = parse_message(
        "https://example.com/ordinary " + long_url,
        None,
        None,
    )
    assert parsed.domains == ["example.com"]
    assert parsed.url_domains == ["example.com", "example.com"]

    analysis = url_signals(parsed)
    items = analysis["urls"]
    assert len(items) == 2
    assert items[0]["suspicious_structure"] is False
    assert items[1]["suspicious_structure"] is True


def test_parser_does_not_treat_email_domain_as_link():
    parsed = parse_message("Email help@example.com for support", None, None)
    assert parsed.urls == []


def test_parser_deduplicates_exact_url_but_preserves_different_paths():
    parsed = parse_message("example.com/a example.com/a example.com/b", None, None)
    assert parsed.urls == ["example.com/a", "example.com/b"]


def test_normalizer_does_not_change_raw():
    raw = "OTP  fr33!!!"
    normalized = normalize_message(raw)
    assert normalized.normalized != raw
    assert normalized.leetspeak


def test_normalizer_preserves_url_case_and_does_not_decode_url_path_as_leetspeak():
    raw = "Please upd8 your account at https://Bank.example/ODFroNwq3J"
    result = normalize_message(raw)

    assert result.normalized == "please update your account at https://Bank.example/ODFroNwq3J"
    # "upd8" is a known abbreviation correction, not a character-by-character leet substitution.
    assert result.spelling_corrected is True
    assert result.leetspeak is False
    assert "odfronwq3j -> odfronwqej" not in result.misspellings


def test_normalizer_does_not_flag_only_a_numeric_url_path_as_leetspeak():
    raw = "Home loan offer expires soon. bnkbzr.co/ODFroNwq3J"
    result = normalize_message(raw)

    assert result.normalized == "home loan offer expires soon. bnkbzr.co/ODFroNwq3J"
    assert result.leetspeak is False
    assert result.misspellings == []


def test_obfuscation_score_does_not_double_count_one_mutation():
    from app.fusion import _obfuscation_risk

    known_abbreviation = normalize_message("blck")
    actual_leetspeak = normalize_message("acc0unt")
    repeated_characters = normalize_message("verifyyyyy")

    assert known_abbreviation.spelling_corrected is True
    assert known_abbreviation.leetspeak is False
    assert _obfuscation_risk(known_abbreviation) == 4.0

    # acc0unt is both a known correction and a leet substitution, but one
    # token mutation must not automatically add both the 4- and 3-point weights.
    assert actual_leetspeak.spelling_corrected is True
    assert actual_leetspeak.leetspeak is True
    assert _obfuscation_risk(actual_leetspeak) == 4.0
    assert repeated_characters.repeated_chars is True
    assert _obfuscation_risk(repeated_characters) == 1.0


def test_suspicious_similarity_is_diagnostic_only_by_default(monkeypatch):
    import asyncio
    from dataclasses import replace

    import app.service as service_module
    from app.service import AnalysisService

    monkeypatch.setattr(
        service_module,
        "settings",
        replace(
            service_module.settings,
            similarity_scoring_enabled=False,
            similarity_timeout_seconds=0.1,
            url_reputation_timeout_seconds=0.1,
        ),
    )
    service = AnalysisService()

    async def fake_classify(message, sender):
        return {
            "model": "SMS-Shield",
            "model_id": "test-model",
            "risk": 70,
            "verdict": "dangerous",
            "model_verdict": "dangerous",
            "verdict_supported": True,
            "verdict_consistent": True,
            "category": "phishing",
            "scam_type": "phishing",
            "red_flags": ["credential request"],
            "explanation": "Credential request.",
            "segments_analyzed": 1,
            "aggregation": "single_message",
            "segment_risks": [70],
        }

    async def no_url_lookup(urls):
        return []

    async def fake_similarity(raw, normalized, ollama):
        return {
            "available": True,
            "matches": [{"id": 1, "distance": 0.01}],
            "top_distance": 0.01,
            "similarity_signal": 5.0,
            "detail": "Strong nearest-neighbour match in demo index.",
        }

    monkeypatch.setattr(service.sms_shield, "classify", fake_classify)
    monkeypatch.setattr(service.url_reputation, "check_urls", no_url_lookup)
    monkeypatch.setattr(service.similarity, "search", fake_similarity)

    result = asyncio.run(
        service.analyze("Please verify your account.", None, post_hoc=False)
    )

    assert result["similarity"]["available"] is True
    assert result["similarity"]["similarity_signal"] == 5.0
    assert result["similarity"]["scoring_enabled"] is False
    assert result["similarity"]["score_contribution"] == 0.0
    assert "diagnostic-only" in result["similarity"]["detail"]
    assert result["evidence"]["similarity"] == 0.0


def test_combination_rule_is_stronger_than_single_keyword():
    signals = behavior_signals("Your KYC is urgent and your account will be blocked. Click https://example.com")
    assert "KYC plus account threat" in signals.combination_hits


def test_credential_safety_warning_is_not_a_request():
    assert not _credential_request("Never share your OTP with anyone.")
    assert not _credential_request("Don’t share your OTP with anyone.")
    assert not _credential_request("Don‘t share your PIN.")
    assert not _credential_request("OTP mat bhejo")


def test_short_safety_warning_does_not_suppress_a_second_request():
    assert _credential_request("Never share OTP. Send PIN.")
    assert _credential_request("OTP mat bhejo, PIN send karo")


def test_hinglish_negation_still_blocks_the_negated_action():
    assert not _credential_request("OTP mat bhejo")
    assert not _credential_request("PIN matt batao")


def test_negated_pair_does_not_suppress_a_separate_positive_request():
    assert _credential_request("Never share your OTP. Please send your PIN to verify.")


def test_group_caps_and_short_message_guard():
    parsed = parse_message("OTP", None, None)
    normalized = normalize_message(parsed.raw)
    result = fuse(parsed, normalized, behavior_signals(parsed.raw), 30, 5)
    assert result.score <= 70
    assert result.evidence["content"] <= 30
    assert result.evidence["similarity"] <= 5



def test_kyc_combination_recognizes_bare_domain_links():
    signals = behavior_signals("Your KYC is incomplete. Update now at bit.ly/verify-account")
    assert "KYC plus URL" in signals.combination_hits


def test_shortened_link_is_not_mislabeled_as_explicit_brand_domain_mismatch():
    from app.rules import url_signals

    parsed = parse_message("SBI KYC reminder: update at bit.ly/verify-account", None, None)
    item = url_signals(parsed)["urls"][0]

    assert item["shortened"] is True
    assert item["brand_domain_mismatch"] is False
    assert item["mismatched_brands"] == []


def test_explicit_conflicting_brand_names_in_message_and_host_are_flagged():
    from app.rules import url_signals

    parsed = parse_message("SBI KYC alert: verify at https://hdfc.example/login", None, None)
    item = url_signals(parsed)["urls"][0]

    assert item["brand_domain_mismatch"] is True
    assert item["mismatched_brands"] == ["sbi"]

def test_slow_optional_similarity_does_not_block_primary_result(monkeypatch):
    import asyncio
    from dataclasses import replace

    import app.service as service_module
    from app.service import AnalysisService

    monkeypatch.setattr(
        service_module,
        "settings",
        replace(
            service_module.settings,
            similarity_timeout_seconds=0.01,
            url_reputation_timeout_seconds=0.1,
        ),
    )
    service = AnalysisService()

    async def fake_classify(message, sender):
        return {
            "model": "SMS-Shield",
            "model_id": "test-model",
            "risk": 15,
            "verdict": "safe",
            "model_verdict": "safe",
            "verdict_supported": True,
            "verdict_consistent": True,
            "category": "personal",
            "scam_type": None,
            "red_flags": [],
            "explanation": "An ordinary personal message.",
            "segments_analyzed": 1,
            "aggregation": "single_message",
            "segment_risks": [15],
        }

    async def no_url_lookup(urls):
        return []

    async def slow_similarity(raw, normalized, ollama):
        await asyncio.sleep(0.05)
        return {
            "available": True,
            "matches": [],
            "top_distance": None,
            "similarity_signal": 0.0,
            "detail": "test",
        }

    async def no_challenger(message):
        return None

    monkeypatch.setattr(service.sms_shield, "classify", fake_classify)
    monkeypatch.setattr(service.url_reputation, "check_urls", no_url_lookup)
    monkeypatch.setattr(service.similarity, "search", slow_similarity)
    monkeypatch.setattr(service.hf, "challenger_signal", no_challenger)

    result = asyncio.run(
        service.analyze("Just saying hello.", None, post_hoc=False)
    )

    assert result["model_assessment"]["risk"] == 15
    assert result["similarity"]["available"] is False
    assert "timed out" in result["similarity"]["detail"]

def test_primary_classifier_timeout_returns_actionable_model_error(monkeypatch):
    import asyncio
    from dataclasses import replace

    import app.service as service_module
    from app.models import ModelUnavailableError
    from app.service import AnalysisService

    monkeypatch.setattr(
        service_module,
        "settings",
        replace(
            service_module.settings,
            classifier_timeout_seconds=0.01,
            url_reputation_timeout_seconds=0.1,
        ),
    )
    service = AnalysisService()

    async def no_url_lookup(urls):
        return []

    async def slow_classify(message, sender):
        await asyncio.sleep(0.05)
        return {}

    monkeypatch.setattr(service.url_reputation, "check_urls", no_url_lookup)
    monkeypatch.setattr(service.sms_shield, "classify", slow_classify)

    try:
        asyncio.run(service.analyze("A short sample SMS.", None, post_hoc=False))
    except ModelUnavailableError as exc:
        assert "0.01-second deadline" in str(exc)
    else:
        raise AssertionError("A timed-out primary classifier must not return a fabricated result")

