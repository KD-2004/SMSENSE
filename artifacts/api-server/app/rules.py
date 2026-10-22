from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .parser import ParsedMessage


# ============================================================
# HINGLISH / ROMAN-HINDI VOCABULARY
# ============================================================

# Used for language/context detection only.
# Hinglish itself does NOT add risk.
STRONG_HINGLISH_WORDS: set[str] = {
    "aap",
    "aapka",
    "aapki",
    "aapke",
    "aapko",
    "aapne",
    "apka",
    "apki",
    "apke",
    "apko",
    "apne",

    "karo",
    "kro",
    "kr",
    "kar",
    "karna",
    "karni",
    "karta",
    "karte",
    "krna",
    "krni",

    "bhejo",
    "bhej",
    "bhejna",

    "batao",
    "btao",
    "bata",
    "bta",
    "dikhao",

    "do",
    "de",
    "dijiye",

    "warna",
    "varna",
    "wrna",

    "hoga",
    "hogi",
    "honge",
    "hai",
    "hain",
    "ho",
    "tha",
    "thi",
    "the",
    "raha",
    "rahe",
    "rahi",

    "abhi",
    "turant",
    "jaldi",
    "foran",
    "fauran",

    "nahi",
    "nahin",
    "mat",
    "matt",

    "kal",
    "aaj",
    "bhai",
    "yaar",
    "sir",
    "madam",

    "paise",
    "paisa",
    "rupaye",
    "rupees",

    "milega",
    "milenge",
    "jeeto",
    "jeet",
    "jeeta",
    "inaam",

    "milna",
    "lena",
    "dena",
    "jana",
    "aana",

    "chahiye",
    "kya",
    "kyu",
    "kaise",
}


# Security/context terms useful when identifying Roman-Hinglish
# security-related messages.
HINDI_SECURITY_TERMS: set[str] = {
    "khata",
    "khaata",
    "band",
    "bandh",
    "rok",
    "roko",
    "chalu",
    "chaalu",
    "suspend",
    "verify",
    "update",
    "activate",
    "deactivate",
}


# ============================================================
# CREDENTIAL DETECTION VOCABULARY
# ============================================================

# Things a message may ask the user to reveal.
CREDENTIAL_TERMS: set[str] = {
    "otp",
    "0tp",
    "pin",
    "cvv",
    "password",
    "passcode",
    "pass",
}


# Actions that can indicate requesting a secret.
REQUEST_ACTIONS: set[str] = {
    "send",
    "provide",
    "tell",
    "give",
    "enter",
    "submit",
    "confirm",
    "share",
    "disclose",
    "reveal",
    "forward",

    # Hinglish
    "bhejo",
    "bhej",
    "batao",
    "btao",
    "batado",
    "dijiye",
}


# Negation / safety words.
NEGATION_SINGLE_WORDS: set[str] = {
    "never",
    "dont",
    "don't",
    "avoid",
    "mat",
    "matt",
}


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass(frozen=True)
class BehaviorSignals:
    urgency: bool
    threats: bool
    kyc: bool
    otp: bool
    upi: bool
    pin: bool
    cvv: bool
    password: bool
    reward: bool
    authority: bool
    requests_secret: bool
    combination_hits: list[str]


# ============================================================
# BASIC HELPERS
# ============================================================

def _has(text: str, pattern: str) -> bool:
    """
    Run a case-insensitive regex.

    VERBOSE is enabled because security expressions below
    are formatted over multiple lines.
    """
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE | re.VERBOSE,
        )
    )


def _tokens(text: str) -> list[str]:
    """
    Return normalized alphanumeric tokens.

    Example:
        "Do not share your OTP!"
        ->
        ["do", "not", "share", "your", "otp"]
    """
    return re.findall(
        r"[a-zA-Z0-9']+",
        (text or "").lower(),
    )


def _normalized_tokens(text: str) -> list[str]:
    """
    Normalize common apostrophe forms in tokens.
    """
    tokens = _tokens(text)

    normalized: list[str] = []

    for token in tokens:
        token = (
            token
            .replace("’", "'")
            .replace("‘", "'")
        )

        normalized.append(token)

    return normalized


# ============================================================
# CREDENTIAL REQUEST ANALYSIS
# ============================================================

def _credential_request(text: str) -> bool:
    """
    Determine whether a message actually asks the recipient
    to reveal/send/enter a credential.

    This intentionally uses token positions rather than one
    giant regex because:

        "Send me your OTP"
        "OTP bhejo"

    are requests, while:

        "Do not share your OTP"
        "Never share your PIN"

    are safety instructions.

    NEGATION HAS PRIORITY.
    """

    tokens = _normalized_tokens(text)

    if not tokens:
        return False

    # --------------------------------------------------------
    # Find credential positions.
    # --------------------------------------------------------

    credential_positions: list[int] = []

    for index, token in enumerate(tokens):
        if token in CREDENTIAL_TERMS:
            credential_positions.append(index)

    # "upi pin" is a compound credential.
    for index in range(len(tokens) - 1):
        if (
            tokens[index] == "upi"
            and tokens[index + 1] == "pin"
        ):
            credential_positions.extend(
                [index, index + 1]
            )

    if not credential_positions:
        return False

    # --------------------------------------------------------
    # Find request-action positions.
    # --------------------------------------------------------

    action_positions: list[int] = [
        index
        for index, token in enumerate(tokens)
        if token in REQUEST_ACTIONS
    ]

    if not action_positions:
        return False

    # --------------------------------------------------------
    # Find explicit negation positions.
    #
    # We also handle:
    # "do not"
    # "please do not"
    # "do not ever"
    # "never ever"
    # --------------------------------------------------------

    negation_positions: list[int] = []

    for index, token in enumerate(tokens):

        if token in NEGATION_SINGLE_WORDS:
            negation_positions.append(index)

        # "do not"
        if (
            token == "do"
            and index + 1 < len(tokens)
            and tokens[index + 1] == "not"
        ):
            negation_positions.extend(
                [index, index + 1]
            )

    # --------------------------------------------------------
    # Determine whether an action and credential are related.
    #
    # Eight tokens is deliberately local enough to avoid
    # unrelated "share" and "OTP" appearing far apart.
    # --------------------------------------------------------

    MAX_DISTANCE = 8

    related_pairs: list[tuple[int, int]] = []

    for action_index in action_positions:
        for credential_index in credential_positions:

            if abs(
                action_index - credential_index
            ) <= MAX_DISTANCE:
                related_pairs.append(
                    (action_index, credential_index)
                )

    if not related_pairs:
        return False

    # Negation applies to a specific nearby action/credential pair only.
    # Continue checking other pairs: a message may contain both a safety
    # warning and a separate request for another secret.
    negated_pairs: set[tuple[int, int]] = set()
    for action_index, credential_index in related_pairs:
        lower = min(action_index, credential_index)
        upper = max(action_index, credential_index)
        if any(lower - 3 <= n <= upper + 3 for n in negation_positions):
            negated_pairs.add((action_index, credential_index))

    # --------------------------------------------------------
    # Roman-Hinglish explicit safety pattern:
    #
    # "OTP mat bhejo"
    # "PIN mat batao"
    # "OTP matt bhej"
    # --------------------------------------------------------

    for credential_index in credential_positions:
        start = max(0, credential_index - 5)
        end = min(len(tokens), credential_index + 6)
        window = tokens[start:end]

        if "mat" in window or "matt" in window:
            for action_index, action_token in enumerate(tokens):
                if action_token in REQUEST_ACTIONS and abs(action_index - credential_index) <= MAX_DISTANCE:
                    negated_pairs.add((action_index, credential_index))

    # --------------------------------------------------------
    # At this point, we have:
    #
    # credential + nearby request action
    # AND no nearby negation.
    #
    # Therefore treat it as an actual request.
    # --------------------------------------------------------

    return any(pair not in negated_pairs for pair in related_pairs)


# ============================================================
# BEHAVIOR / SOCIAL ENGINEERING
# ============================================================

def behavior_signals(
    text: str,
) -> BehaviorSignals:
    """
    Detect phishing/social-engineering evidence.

    This function does NOT calculate the final risk score.
    Final scoring remains in fusion.py.
    """

    value = (
        text or ""
    ).strip().lower()

    # --------------------------------------------------------
    # 1. URGENCY
    # --------------------------------------------------------

    urgency = _has(
        value,
        r"""
        \b(?:
            urgent
            | immediately
            | immediate
            | right\s*now
            | as\s+soon\s+as\s+possible
            | within\s+\d+
            | today
            | now
            | last\s+warning
            | final\s+warning
            | abhi
            | turant
            | jaldi
            | foran
            | fauran
        )\b
        |
        \babhi\s+(?:karo|kr|kro|bhejo)\b
        |
        \bjaldi\s+(?:karo|kr|kro|bhejo)\b
        |
        \bisi\s+(?:waqt|time)\b
        """,
    )

    # --------------------------------------------------------
    # 2. THREATS
    # --------------------------------------------------------

    threats = _has(
        value,
        r"""
        \bblocked?\b
        |
        \bblck\b
        |
        \bblock\s+hoga\b
        |
        \bblck\s+hoga\b
        |
        \baccount\s+will\s+be\s+blocked\b
        |
        \baccount\s+blocked?\b
        |
        \baccount\s+(?:band|bandh)\b
        |
        \baccount\s+band\s+ho(?:ga|jayega|jaega)\b
        |
        \baccount\s+suspend(?:ed)?\b
        |
        \bsuspend\s+hoga\b
        |
        \bdeactivate(?:d)?\b
        |
        \bdeactivate\s+hoga\b
        |
        \blegal\s+action\b
        |
        \blegal\s+karwai\b
        |
        \barrest\b
        |
        \bpolice\s+action\b
        |
        \bpenalty\b
        |
        \bfine\b
        |
        \bwarna\s+account\b
        |
        \bvarna\s+account\b
        |
        \bwrna\s+account\b
        |
        \baccess\s+(?:will\s+be\s+)?blocked\b
        """,
    )

    # --------------------------------------------------------
    # 3. KYC
    # --------------------------------------------------------

    kyc = _has(
        value,
        r"""
        \bkyc\b
        |
        \bknow\s+your\s+customer\b
        |
        \bre[-\s]?verify\b
        |
        \bkyc\s+(?:update|upd8|updte|updt)\b
        |
        \bkyc\s+(?:karo|kar|kr|kro)\b
        |
        \bkyc\s+(?:complete|submit|verify)\b
        """,
    )

    # --------------------------------------------------------
    # 4. OTP
    # --------------------------------------------------------

    otp = _has(
        value,
        r"""
        \botp\b
        |
        \b0tp\b
        |
        \bone[-\s]?time\s+password\b
        |
        \bone[-\s]?time\s+passcode\b
        """,
    )

    # --------------------------------------------------------
    # 5. UPI
    # --------------------------------------------------------

    upi = _has(
        value,
        r"""
        \bupi\b
        |
        \bvpa\b
        |
        \bupi\s+pin\b
        |
        \bcollect\s+request\b
        """,
    )

    # --------------------------------------------------------
    # 6. PIN
    # --------------------------------------------------------

    pin = _has(
        value,
        r"""
        \bpin\b
        |
        \bupi\s+pin\b
        |
        \batm\s+pin\b
        """,
    )

    # --------------------------------------------------------
    # 7. CVV
    # --------------------------------------------------------

    cvv = _has(
        value,
        r"\bcvv\b",
    )

    # --------------------------------------------------------
    # 8. PASSWORD
    # --------------------------------------------------------

    password = _has(
        value,
        r"""
        \bpassword\b
        |
        \bpasscode\b
        |
        \bpass\s+code\b
        |
        \blogin\s+code\b
        """,
    )

    # --------------------------------------------------------
    # 9. REWARD / PRIZE
    # --------------------------------------------------------

    reward = _has(
        value,
        r"""
        \bprize\b
        |
        \breward\b
        |
        \blottery\b
        |
        \bwinner\b
        |
        \bcashback\b
        |
        \bbonus\b
        |
        \bfree\s+gift\b
        |
        \byou\s+won\b
        |
        \bcongratulations\b
        |
        \binaam\b
        |
        \bjeeto?\b
        |
        \bjeeta\b
        |
        \bmilega\b
        """,
    )

    # --------------------------------------------------------
    # 10. AUTHORITY / BRAND CLAIM
    # --------------------------------------------------------

    authority = _has(
        value,
        r"""
        \bbank\b
        |
        \brbi\b
        |
        \bsbi\b
        |
        \bhdfc\b
        |
        \bicici\b
        |
        \baxis\b
        |
        \bkotak\b
        |
        \bpaytm\b
        |
        \bphonepe\b
        |
        \bgooglepay\b
        |
        \bgpay\b
        |
        \baadhaar\b
        |
        \buidai\b
        |
        \bincome\s+tax\b
        |
        \bgovernment\b
        |
        \bpolice\b
        |
        \bcustoms\b
        |
        \bcourier\b
        |
        \bpost\b
        |
        \btelecom\b
        |
        \bairtel\b
        |
        \bjio\b
        |
        \bvi\b
        """,
    )

    # --------------------------------------------------------
    # 11. ACTUAL CREDENTIAL REQUEST
    # --------------------------------------------------------

    requests_secret = _credential_request(
        value
    )

    # ========================================================
    # COMBINATION RULES
    # ========================================================

    combinations: list[str] = []

    # KYC + account threat.
    if kyc and threats:
        combinations.append(
            "KYC plus account threat"
        )

    # KYC + urgency.
    if kyc and urgency:
        combinations.append(
            "KYC plus urgency"
        )

    # KYC + URL.
    if kyc and _has(
        value,
        r"(?:https?://|www\.)",
    ):
        combinations.append(
            "KYC plus URL"
        )

    # OTP + actual credential request.
    if otp and requests_secret:
        combinations.append(
            "OTP sharing request"
        )

    # Other credential + actual request.
    if (
        (upi or pin or cvv or password)
        and requests_secret
    ):
        combinations.append(
            "credential sharing request"
        )

    # Reward + payment.
    if reward and _has(
        value,
        r"""
        \b(?:
            pay
            | fee
            | charge
            | deposit
            | upi
            | send
            | payment
            | bhejo
            | jama
        )\b
        """,
    ):
        combinations.append(
            "reward plus payment request"
        )

    # Job + payment/registration fee.
    if (
        _has(
            value,
            r"""
            \b(?:
                job
                | work\s+from\s+home
                | hiring
                | naukri
                | job\s+offer
                | employment
            )\b
            """,
        )
        and _has(
            value,
            r"""
            \b(?:
                fee
                | registration
                | pay
                | payment
                | charge
                | deposit
            )\b
            """,
        )
    ):
        combinations.append(
            "job offer plus fee"
        )

    # Authority + KYC.
    if authority and kyc:
        combinations.append(
            "authority plus KYC"
        )

    # Authority + threat.
    if authority and threats:
        combinations.append(
            "authority plus threat"
        )

    return BehaviorSignals(
        urgency=urgency,
        threats=threats,
        kyc=kyc,
        otp=otp,
        upi=upi,
        pin=pin,
        cvv=cvv,
        password=password,
        reward=reward,
        authority=authority,
        requests_secret=requests_secret,
        combination_hits=list(
            dict.fromkeys(combinations)
        ),
    )


# ============================================================
# SENDER ANALYSIS
# ============================================================

def sender_signals(
    sender: Optional[str],
) -> dict[str, object]:
    """
    Sender analysis produces evidence only.

    Missing sender / phone / trusted-brand claim are not
    automatically malicious.
    """

    if not sender or not sender.strip():
        return {
            "type": "missing",
            "is_phone": False,
            "is_header": False,
            "suspicious": False,
            "anomalies": [],
        }

    value = sender.strip()

    # Phone number.
    is_phone = bool(
        re.fullmatch(
            r"\+?[0-9][0-9\s()\-]{6,18}",
            value,
        )
    )

    # Alphanumeric SMS header.
    is_header = (
        bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9\-]{1,15}",
                value,
            )
        )
        and not is_phone
    )

    anomalies: list[str] = []

    # Strange characters.
    if re.search(
        r"[^A-Za-z0-9+\-()\s]",
        value,
    ):
        anomalies.append(
            "unusual sender characters"
        )

    # Promotional sender identifier.
    if re.search(
        r"""
        \b(?:
            free
            | prize
            | winner
            | claim
            | reward
            | bonus
            | cashback
        )\b
        """,
        value,
        re.IGNORECASE | re.VERBOSE,
    ):
        anomalies.append(
            "promotional sender identifier"
        )

    # Trusted-brand / authority claim.
    #
    # This is an anomaly, not proof of spoofing.
    if re.search(
        r"""
        \b(?:
            sbi
            | hdfc
            | icici
            | axis
            | rbi
            | bank
            | uidai
            | aadhaar
        )\b
        """,
        value,
        re.IGNORECASE | re.VERBOSE,
    ):
        anomalies.append(
            "trusted-brand/authority claim in sender"
        )

    return {
        "type": (
            "phone"
            if is_phone
            else "header"
            if is_header
            else "other"
        ),
        "is_phone": is_phone,
        "is_header": is_header,
        "suspicious": bool(anomalies),
        "anomalies": list(
            dict.fromkeys(anomalies)
        ),
    }


# ============================================================
# URL ANALYSIS
# ============================================================

def url_signals(
    parsed: ParsedMessage,
) -> dict[str, object]:
    """
    Structural URL analysis.

    This does NOT claim maliciousness unless an actual
    reputation/intelligence provider returns that evidence.
    """

    results: list[dict[str, object]] = []

    brand_aliases: dict[str, set[str]] = {
        "sbi": {"sbi"},
        "hdfc": {"hdfc"},
        "icici": {"icici"},
        "axis": {"axis"},
        "paytm": {"paytm"},
        "phonepe": {"phonepe"},
        "gpay": {"gpay", "googlepay"},
        "rbi": {"rbi"},
        "uidai": {"uidai"},
        "aadhaar": {"aadhaar"},
    }

    # Brands explicitly claimed in the message.
    claimed_brands: set[str] = set()

    for brand, aliases in brand_aliases.items():
        for alias in aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b",
                parsed.raw,
                re.IGNORECASE,
            ):
                claimed_brands.add(brand)
                break

    for raw_url, domain in zip(
        parsed.urls,
        parsed.domains,
    ):
        lowered_url = raw_url.lower()
        lowered_domain = domain.lower()

        # --------------------------------------------------------
        # Shortened URL
        # --------------------------------------------------------

        shortened = lowered_domain in {
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "is.gd",
            "cutt.ly",
            "rb.gy",
            "ow.ly",
            "shorturl.at",
        }

        # --------------------------------------------------------
        # Structural anomalies
        # --------------------------------------------------------

        suspicious_structure = bool(
            "@" in lowered_url
            or re.search(
                r"[^\x00-\x7f]",
                lowered_domain,
            )
            or lowered_domain.count("-") >= 3
            or bool(
                re.search(
                    r"\d{5,}",
                    lowered_domain,
                )
            )
            or len(lowered_url) >= 180
        )

        # --------------------------------------------------------
        # Direct IP URL
        # --------------------------------------------------------

        ip_address_url = bool(
            re.fullmatch(
                r"""
                (?:https?://)?
                (?:\d{1,3}\.){3}
                \d{1,3}
                (?::\d+)?
                (?:/.*)?
                """,
                lowered_url,
                re.IGNORECASE | re.VERBOSE,
            )
        )

        if ip_address_url:
            suspicious_structure = True

        # --------------------------------------------------------
        # Brand / domain mismatch
        # --------------------------------------------------------

        brand_domain_mismatch = False
        mismatched_brands: list[str] = []

        for brand in sorted(
            claimed_brands
        ):
            brand_present = bool(
                re.search(
                    rf"(^|[.\-]){re.escape(brand)}([.\-]|$)",
                    lowered_domain,
                    re.IGNORECASE,
                )
            )

            if not brand_present:
                brand_domain_mismatch = True
                mismatched_brands.append(
                    brand
                )

        results.append(
            {
                "url": raw_url,
                "domain": domain,
                "shortened": shortened,
                "ip_address_url": ip_address_url,
                "suspicious_structure": suspicious_structure,
                "brand_domain_mismatch": brand_domain_mismatch,
                "mismatched_brands": mismatched_brands,
                "reputation": "not_checked",
            }
        )

    return {
        "urls": results,
        "count": len(results),
        "evidence_basis": (
            "structure-only; reputation and redirects were not checked"
        ),
    }


# ============================================================
# LANGUAGE HINT
# ============================================================

def language_hint(text: str) -> str:
    """
    Lightweight language hint for Indian SMS.

    This is NOT a trained language-identification model.

    Returns:
        hi        = Devanagari Hindi
        hinglish  = likely Roman Hindi / Hinglish
        mixed     = mixed Unicode/Latin
        en        = likely English
        other     = other/non-text
        unknown   = empty input

    Design:
        - A single common English word must never make a message Hinglish.
        - Strong Roman-Hindi markers can identify Hinglish by themselves.
        - Ambiguous words require multiple supporting indicators.
    """

    if not text or not text.strip():
        return "unknown"

    value = text.lower().strip()

    # --------------------------------------------------------
    # 1. Native Hindi / Devanagari
    # --------------------------------------------------------

    if re.search(
        r"[\u0900-\u097F]",
        value,
    ):
        return "hi"

    # --------------------------------------------------------
    # 2. Other non-ASCII
    # --------------------------------------------------------

    if re.search(
        r"[^\x00-\x7F]",
        value,
    ):
        if not re.search(
            r"[a-zA-Z]",
            value,
        ):
            return "other"

        return "mixed"

    # --------------------------------------------------------
    # 3. Tokenize
    # --------------------------------------------------------

    tokens = _tokens(value)

    if not tokens:
        return "unknown"

    token_set = set(tokens)

    # --------------------------------------------------------
    # 4. STRONG Hinglish indicators
    #
    # These are sufficiently characteristic that one is enough.
    # --------------------------------------------------------

    strong_hinglish: set[str] = {
        "aap",
        "aapka",
        "aapki",
        "aapke",
        "aapko",
        "aapne",

        "apka",
        "apki",
        "apke",
        "apko",
        "apne",

        "karo",
        "kro",
        "krna",
        "krni",

        "bhejo",
        "bhejna",

        "batao",
        "btao",
        "batado",
        "bta",

        "warna",
        "varna",
        "wrna",

        "hoga",
        "hogi",
        "honge",

        "hain",

        "abhi",
        "turant",
        "jaldi",
        "foran",
        "fauran",

        "nahi",
        "nahin",

        "paise",
        "paisa",
        "rupaye",

        "milega",
        "milenge",

        "jeeta",
        "jeeto",
        "inaam",

        "bhai",
        "yaar",

        "dijiye",
        "chahiye",

        "kyu",
        "kaise",
    }

    if token_set.intersection(
        strong_hinglish
    ):
        return "hinglish"

    # --------------------------------------------------------
    # 5. Ambiguous / contextual Hinglish indicators
    #
    # These words can also occur in ordinary English.
    # Require at least TWO contextual indicators.
    # --------------------------------------------------------

    contextual_hinglish: set[str] = {
        "kr",
        "kar",
        "karna",
        "karni",
        "karta",
        "karte",

        "ho",
        "hai",
        "tha",
        "thi",
        "the",
        "raha",
        "rahe",
        "rahi",

        "kal",
        "aaj",

        "mat",
        "matt",

        "lena",
        "dena",
        "jana",
        "aana",

        "kya",
    }

    contextual_hits = token_set.intersection(
        contextual_hinglish
    )

    if len(contextual_hits) >= 2:
        return "hinglish"

    # --------------------------------------------------------
    # 6. Security-context combination
    #
    # For messages such as:
    #   "KYC karna hai"
    #
    # Security vocabulary alone does not mean Hinglish.
    # We need multiple Indian-language/context indicators.
    # --------------------------------------------------------

    security_context = {
        "khata",
        "khaata",
        "band",
        "bandh",
        "rok",
        "roko",
        "chalu",
        "chaalu",
    }

    security_hits = token_set.intersection(
        security_context
    )

    if (
        len(security_hits) >= 1
        and len(contextual_hits) >= 1
    ):
        return "hinglish"

    # --------------------------------------------------------
    # 7. Otherwise English
    # --------------------------------------------------------

    return "en"