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


def test_combination_rule_is_stronger_than_single_keyword():
    signals = behavior_signals("Your KYC is urgent and your account will be blocked. Click https://example.com")
    assert "KYC plus account threat" in signals.combination_hits


def test_credential_safety_warning_is_not_a_request():
    assert not _credential_request("Never share your OTP with anyone.")
    assert not _credential_request("OTP mat bhejo")


def test_negated_pair_does_not_suppress_a_separate_positive_request():
    assert _credential_request("Never share your OTP. Please send your PIN to verify.")


def test_group_caps_and_short_message_guard():
    parsed = parse_message("OTP", None, None)
    normalized = normalize_message(parsed.raw)
    result = fuse(parsed, normalized, behavior_signals(parsed.raw), 30, 5)
    assert result.score <= 70
    assert result.evidence["content"] <= 30
    assert result.evidence["similarity"] <= 5
