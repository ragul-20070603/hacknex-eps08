# ============================================================
# HNX26EPS08 - SAFETY FILTER
# ============================================================

BLOCKED_TERMS = [
    "malware",
    "ransomware",
    "keylogger",
    "phishing",
]


SAFE_RESPONSE = (
    "I can't provide instructions for that. "
    "I can help with a safe alternative."
)


def safety_filter(text):
    """
    Check LLM-generated text before sending it to TTS.

    Returns:
        (safe_text, blocked)
    """

    text_lower = text.lower()

    for term in BLOCKED_TERMS:
        if term in text_lower:
            return SAFE_RESPONSE, True

    return text, False