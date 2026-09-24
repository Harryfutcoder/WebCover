"""Small fallback for exrex.getone used for regex-like input generation."""


def getone(pattern, *args, **kwargs):
    text = str(pattern or "")
    # Prefer a harmless alphanumeric token over trying to implement regex generation.
    return "test" if text else ""
