from app.fusion import fuse
from app.normalizer import normalize_message
from app.parser import parse_message
from app.rules import behavior_signals


def test_parser_preserves_raw_and_extracts_url():
    raw = "  KYC!!! visit https://bit.ly/abc  "
    parsed = parse_message(raw, "BANKALERT", None)
    assert parsed.raw == raw
    assert parsed.domains == ["bit.ly"]
    assert parsed.length == len(raw)


def test_normalizer_does_not_change_raw():
    raw = "OTP  fr33!!!"
    normalized = normalize_message(raw)
    assert normalized.normalized != raw
    assert normalized.leetspeak


def test_combination_rule_is_stronger_than_single_keyword():
    signals = behavior_signals("Your KYC is urgent and your account will be blocked. Click https://example.com")
    assert "KYC plus account threat" in signals.combination_hits


def test_group_caps_and_short_message_guard():
    parsed = parse_message("OTP", None, None)
    normalized = normalize_message(parsed.raw)
    result = fuse(parsed, normalized, behavior_signals(parsed.raw), 30, 5)
    assert result.score <= 70
    assert result.evidence["content"] <= 30
    assert result.evidence["similarity"] <= 5