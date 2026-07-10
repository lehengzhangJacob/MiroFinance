"""
Text helpers for commit message formatting.
"""

import re


def slugify(text: str) -> str:
    """Lowercase, replace spaces with hyphens, remove illegal chars, collapse repeated hyphens."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)  # Collapse repeated hyphens
    return text


def truncate_subject(line: str, max_len: int = 72) -> str:
    """Truncate a commit subject line to max_len characters."""
    line = line.strip()
    if len(line) <= max_len:
        return line
    return line[:max_len].rstrip()  # Fix off-by-one error