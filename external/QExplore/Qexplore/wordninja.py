"""Conservative fallback for wordninja.split."""

import re


def split(text):
    text = str(text or "")
    chunks = re.findall(r"[A-Za-z]+|\d+", text)
    return chunks or ([text] if text else [])
