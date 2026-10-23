from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import URL_RE


# ============================================================
# 1. KNOWN CHALLENGE OBFUSCATIONS
# ============================================================
#
# These are checked BEFORE generic leetspeak.
# This is important: "upd8" must become "update",
# not "updb".
#
KNOWN_OBFUSCATIONS: dict[str, str] = {
    "upd8": "update",
    "updt": "update",
    "updte": "update",

    "blck": "block",
    "blk": "block",

    "accnt": "account",
    "acnt": "account",
    "acc0unt": "account",

    "cl1ck": "click",
    "clck": "click",

    "ver1fy": "verify",
    "v3rify": "verify",
    "vrify": "verify",
    "verfy": "verify",

    "0tp": "otp",

    "p4ssword": "password",
    "passw0rd": "password",
    "pa55word": "password",

    "s3cur1ty": "security",
    "secur1ty": "security",

    "f4c3b00k": "facebook",
    "faceb00k": "facebook",

    "wh4ts4pp": "whatsapp",
    "watsapp": "whatsapp",

    "t3l3gram": "telegram",
    "teligram": "telegram",
}


# ============================================================
# 2. COMMON TYPOS
# ============================================================

COMMON_MISSPELLINGS: dict[str, str] = {
    "acount": "account",
    "suspand": "suspend",
    "confrim": "confirm",
    "recieve": "receive",
    "imediately": "immediately",
    "lotery": "lottery",
    "verfiy": "verify",
    "seperate": "separate",
    "definately": "definitely",
    "occured": "occurred",
    "untill": "until",
    "becuase": "because",
    "whch": "which",
}


# ============================================================
# 3. ROMAN HINDI / HINGLISH WORDS
# ============================================================

HINGLISH_WORDS: set[str] = {
    "aap",
    "aapka",
    "aapki",
    "aapke",
    "aapko",
    "apka",
    "apki",
    "apke",
    "apko",

    "karo",
    "kro",
    "kr",
    "kar",
    "karna",
    "karni",
    "karte",
    "karta",
    "krna",
    "krni",

    "hoga",
    "hogi",
    "honge",
    "hai",
    "hain",
    "ho",
    "tha",
    "thi",
    "the",

    "warna",
    "varna",
    "wrna",

    "abhi",
    "turant",
    "jaldi",
    "foran",
    "fauran",

    "bhejo",
    "bhej",
    "bhejna",

    "batao",
    "btao",
    "bata",
    "bta",

    "milega",
    "milenge",
    "jeeta",
    "jeeto",
    "inaam",

    "paise",
    "paisa",
    "rupaye",

    "nahi",
    "nahin",
    "mat",

    "bhai",
    "yaar",

    "aaj",
    "kal",

    "jana",
    "jana",
    "aana",
    "lena",
    "dena",
    "milna",

    "chahiye",
    "kyu",
    "kya",
    "kaise",
}


# ============================================================
# 4. SAFE LEETSPEAK
# ============================================================
#
# IMPORTANT:
# We DO NOT globally translate every number.
#
# 4821 must stay 4821.
# 10 must stay 10.
#
# We only run leetspeak decoding if the token contains:
#   - at least one alphabetic character
#   - and at least one substitution character
#
# Also, only conservative mappings are used.
#
SAFE_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})


# ============================================================
# 5. RESULT
# ============================================================

@dataclass(frozen=True)
class NormalizationResult:
    normalized: str
    transliteration: bool
    misspellings: list[str]
    leetspeak: bool
    punctuation_manipulation: bool
    spacing_manipulation: bool
    repeated_chars: bool
    concatenated_split: bool
    spelling_corrected: bool


# ============================================================
# 6. HELPERS
# ============================================================

TOKEN_RE = re.compile(r"[A-Za-z0-9@$]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _leet_decode(token: str) -> tuple[str, bool]:
    """
    Decode conservative leetspeak only when the token contains
    both letters and substitution characters.

    Examples:
        upd8    -> updt   (generic step; known mapping runs first)
        acc0unt  -> account
        4821     -> unchanged
        10       -> unchanged
        999      -> unchanged
    """

    if not re.search(r"[a-z]", token):
        return token, False

    if not re.search(r"[013457@$]", token):
        return token, False

    decoded = token.translate(SAFE_LEET_MAP)

    return decoded, decoded != token


def _reduce_repeated_chars(token: str) -> tuple[str, bool]:
    """
    Reduce exaggerated repeated characters.

    "verifyyyyy" -> "verifyy"

    This is only an obfuscation indicator.
    It should not aggressively rewrite ordinary words.
    """

    reduced = re.sub(r"(.)\1{2,}", r"\1\1", token)

    return reduced, reduced != token


def _detect_punctuation_manipulation(raw: str) -> bool:
    return bool(
        re.search(r"[!?]{2,}", raw)
        or re.search(r"\.{3,}", raw)
        or re.search(r"[^\w\s]{3,}", raw)
    )


def _detect_spacing_manipulation(raw: str) -> bool:
    lowered = raw.lower()

    return bool(
        re.search(r"\bo\s+t\s+p\b", lowered)
        or re.search(r"\bp\s+i\s+n\b", lowered)
        or re.search(r"\bk\s+y\s+c\b", lowered)
        or re.search(r"\w\s{2,}\w", raw)
    )


def _detect_hinglish(tokens: list[str]) -> bool:
    return bool(set(tokens).intersection(HINGLISH_WORDS))


# ============================================================
# 7. NORMALIZATION
# ============================================================

def normalize_message(raw: str) -> NormalizationResult:
    """
    Create a secondary normalized representation without rewriting URLs.

    Raw SMS is never modified or discarded. URL spans are preserved exactly in
    the normalized copy so numeric URL paths are not misclassified as leetspeak
    and callers never receive a mutated destination from this representation.
    """

    raw = raw or ""
    detected_changes: list[str] = []
    leet_detected = False
    repeated_detected = False
    spelling_detected = False

    def normalize_plain_text(segment: str) -> str:
        nonlocal leet_detected, repeated_detected, spelling_detected
        lowered = segment.lower()
        original_tokens = _tokens(lowered)
        normalized_tokens: list[str] = []

        for original in original_tokens:
            token = original

            if token in KNOWN_OBFUSCATIONS:
                replacement = KNOWN_OBFUSCATIONS[token]
                normalized_tokens.append(replacement)
                detected_changes.append(f"{token} -> {replacement}")
                spelling_detected = True
                # Some known obfuscations are abbreviations (for example, "blck"),
                # not leetspeak. Mark leetspeak only when a substitution actually
                # occurs; fusion must not treat one mapping as two independent clues.
                _decoded, is_leet = _leet_decode(original)
                leet_detected = leet_detected or is_leet
                continue

            if token in COMMON_MISSPELLINGS:
                replacement = COMMON_MISSPELLINGS[token]
                normalized_tokens.append(replacement)
                detected_changes.append(f"{token} -> {replacement}")
                spelling_detected = True
                continue

            decoded, changed = _leet_decode(token)
            if changed:
                normalized_tokens.append(decoded)
                detected_changes.append(f"{token} -> {decoded}")
                leet_detected = True
                continue

            reduced, repeated = _reduce_repeated_chars(token)
            if repeated:
                normalized_tokens.append(reduced)
                detected_changes.append(f"{token} -> {reduced}")
                repeated_detected = True
                continue

            normalized_tokens.append(token)

        replacement_queue = iter(normalized_tokens)

        def replace_match(match: re.Match[str]) -> str:
            try:
                return next(replacement_queue)
            except StopIteration:
                return match.group(0)

        return TOKEN_RE.sub(replace_match, lowered)

    normalized_parts: list[str] = []
    cursor = 0
    for url_match in URL_RE.finditer(raw):
        normalized_parts.append(normalize_plain_text(raw[cursor:url_match.start()]))
        normalized_parts.append(url_match.group(0))
        cursor = url_match.end()
    normalized_parts.append(normalize_plain_text(raw[cursor:]))

    normalized = re.sub(r"\s+", " ", "".join(normalized_parts)).strip()
    message_without_urls = URL_RE.sub(" ", raw)
    content_tokens = _tokens(message_without_urls.lower())

    return NormalizationResult(
        normalized=normalized,
        transliteration=_detect_hinglish(content_tokens),
        misspellings=list(dict.fromkeys(detected_changes)),
        leetspeak=leet_detected,
        punctuation_manipulation=_detect_punctuation_manipulation(message_without_urls),
        spacing_manipulation=_detect_spacing_manipulation(message_without_urls),
        repeated_chars=repeated_detected,
        concatenated_split=False,
        spelling_corrected=spelling_detected,
    )
