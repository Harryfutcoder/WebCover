import importlib
import os
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def instantiate_class_by_module_and_class_name(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz()
    return instance


def get_class_by_module_and_class_name(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_class_by_module_and_class_name_and_params(module_name: str, class_name: str, params) -> Any:
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz(params)
    return instance


def safe_console_text(value: Any) -> str:
    """Convert any value to an ASCII-safe printable string for GBK/legacy consoles."""
    text = str(value)
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def normalize_url_for_state(raw_url: str, mode: Optional[str] = None) -> str:
    """
    Normalize URL for state identity / redirect identity.

    Modes:
      - path_only: keep scheme+host+path, drop query+fragment.
      - strip_tracking: keep query except common tracking/auth redirect params.
      - raw: keep query.

    By default fragments are dropped for backward compatibility. SPA benchmarks
    can opt in to preserving hash routes with WEBTEST_URL_PRESERVE_HASH_ROUTE=1.
    """
    if not raw_url:
        return ""

    resolved_mode = (
        mode
        or os.environ.get("WEBTEST_URL_NORMALIZE_MODE", "strip_tracking")
    ).strip().lower()

    try:
        parsed = urlsplit(raw_url)
        scheme = parsed.scheme
        netloc = parsed.netloc
        path = parsed.path or "/"

        if resolved_mode in ("path_only", "strict", "drop_query"):
            query = ""
        elif resolved_mode in ("raw", "keep_query", "none"):
            query = parsed.query
        else:
            # Backward-compatible mode used by older runs.
            auth_path = (
                path.startswith("/login")
                or path.startswith("/signup")
                or path.startswith("/password_reset")
            )
            if auth_path:
                query = ""
            else:
                kept = []
                for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                    kl = k.lower()
                    if kl.startswith("utm_") or kl.startswith("ref_") or kl in {"source", "return_to"}:
                        continue
                    kept.append((k, v))
                query = urlencode(kept, doseq=True)

        preserve_hash_route = os.environ.get(
            "WEBTEST_URL_PRESERVE_HASH_ROUTE",
            "0",
        ).strip().lower() in ("1", "true", "yes", "on")
        fragment = parsed.fragment if preserve_hash_route else ""
        return urlunsplit((scheme, netloc, path, query, fragment))
    except Exception:
        return raw_url.split("?", 1)[0]


def read_reward_clip_abs(default: float = 5.0) -> Optional[float]:
    raw_clip_abs = os.environ.get("WEBTEST_REWARD_CLIP_ABS", str(default))
    try:
        reward_clip_abs = float(raw_clip_abs)
    except (TypeError, ValueError):
        reward_clip_abs = float(default)
    return reward_clip_abs if reward_clip_abs > 0 else None


def clip_reward_value(value: float, clip_abs: Optional[float]) -> float:
    if clip_abs is None:
        return float(value)
    return float(max(-clip_abs, min(clip_abs, float(value))))
