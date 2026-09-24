"""Small local fallback for the cryptohash package used by QExplore."""

import hashlib


def md5(value):
    if isinstance(value, bytes):
        data = value
    else:
        data = str(value).encode("utf-8", errors="ignore")
    return hashlib.md5(data).hexdigest()
