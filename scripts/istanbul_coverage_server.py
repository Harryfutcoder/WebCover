"""Small Istanbul coverage listener compatible with WebRLED's /coverage/client.

The original WebRLED artifact expects an express-istanbul service that accepts
browser-side ``window.__coverage__`` at ``POST /coverage/client`` and serves an
HTML summary at ``GET /coverage``. This script implements the minimal contract
with only Python's standard library, so experiments do not depend on a missing
``express-istanbul.zip`` bundle.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


Coverage = dict[str, Any]


def _merge_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key, value in (src or {}).items():
        try:
            dst[key] = int(dst.get(key, 0)) + int(value)
        except (TypeError, ValueError):
            continue


def _merge_branch_counts(dst: dict[str, list[int]], src: dict[str, Any]) -> None:
    for key, values in (src or {}).items():
        if not isinstance(values, list):
            continue
        existing = dst.setdefault(key, [0] * len(values))
        if len(existing) < len(values):
            existing.extend([0] * (len(values) - len(existing)))
        for idx, value in enumerate(values):
            try:
                existing[idx] += int(value)
            except (TypeError, ValueError):
                continue


def merge_coverage(dst: Coverage, src: Coverage | None) -> Coverage:
    if not isinstance(src, dict):
        return dst
    for filename, file_cov in src.items():
        if not isinstance(file_cov, dict):
            continue
        target = dst.setdefault(filename, {})
        for meta_key in ("path", "statementMap", "fnMap", "branchMap"):
            if meta_key in file_cov:
                target[meta_key] = file_cov[meta_key]
        _merge_counts(target.setdefault("s", {}), file_cov.get("s") or {})
        _merge_counts(target.setdefault("f", {}), file_cov.get("f") or {})
        _merge_branch_counts(target.setdefault("b", {}), file_cov.get("b") or {})
    return dst


def _fraction(covered: int, total: int) -> tuple[str, str]:
    if total <= 0:
        return "100.00%", "0/0"
    return f"{covered * 100.0 / total:.2f}%", f"{covered}/{total}"


def summarize(coverage: Coverage) -> dict[str, tuple[str, str]]:
    statements_total = statements_covered = 0
    functions_total = functions_covered = 0
    branches_total = branches_covered = 0
    line_hits: dict[tuple[str, int], int] = {}

    for filename, file_cov in coverage.items():
        if not isinstance(file_cov, dict):
            continue

        statement_counts = file_cov.get("s") or {}
        statement_map = file_cov.get("statementMap") or {}
        statements_total += len(statement_counts)
        statements_covered += sum(1 for count in statement_counts.values() if int(count or 0) > 0)
        for stmt_id, loc in statement_map.items():
            start = (loc or {}).get("start") or {}
            line = start.get("line")
            if line is None:
                continue
            try:
                hit = int(statement_counts.get(stmt_id, 0) or 0)
                line_hits[(str(filename), int(line))] = max(line_hits.get((str(filename), int(line)), 0), hit)
            except (TypeError, ValueError):
                continue

        function_counts = file_cov.get("f") or {}
        functions_total += len(function_counts)
        functions_covered += sum(1 for count in function_counts.values() if int(count or 0) > 0)

        branch_counts = file_cov.get("b") or {}
        for values in branch_counts.values():
            if not isinstance(values, list):
                continue
            branches_total += len(values)
            branches_covered += sum(1 for count in values if int(count or 0) > 0)

    lines_total = len(line_hits)
    lines_covered = sum(1 for hit in line_hits.values() if hit > 0)
    return {
        "Statements": _fraction(statements_covered, statements_total),
        "Branches": _fraction(branches_covered, branches_total),
        "Functions": _fraction(functions_covered, functions_total),
        "Lines": _fraction(lines_covered, lines_total),
    }


def make_html(summary: dict[str, tuple[str, str]]) -> str:
    rows = []
    for label in ("Statements", "Branches", "Functions", "Lines"):
        percentage, fraction = summary[label]
        rows.append(
            "<div class='metric'>"
            f"<span class=\"strong\">{percentage} </span> "
            f"<span class=\"quiet\">{label}</span> "
            f"<span class='fraction'>{fraction}</span>"
            "</div>"
        )
    return "<!doctype html><html><body>" + "\n".join(rows) + "</body></html>"


class CoverageHandler(BaseHTTPRequestHandler):
    coverage: Coverage = {}
    out_dir: Path = Path("analysis/webrled_code_coverage/istanbul_listener")
    log_file: Path | None = None

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("", "/coverage"):
            self._send(404, b"not found", "text/plain")
            return
        html = make_html(summarize(self.coverage)).encode("utf-8")
        self._send(200, html, "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in ("/coverage/reset", "/reset"):
            self.coverage = {}
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "coverage.json").write_text("{}", encoding="utf-8")
            (self.out_dir / "index.html").write_text(make_html(summarize(self.coverage)), encoding="utf-8")
            self.log_message("reset coverage")
            self._send(200, b'{"ok":true,"reset":true}', "application/json")
            return

        if path != "/coverage/client":
            self._send(404, b"not found", "text/plain")
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except json.JSONDecodeError:
            self._send(400, b"invalid json", "text/plain")
            return
        merge_coverage(self.coverage, payload)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "coverage.json").write_text(json.dumps(self.coverage), encoding="utf-8")
        (self.out_dir / "index.html").write_text(make_html(summarize(self.coverage)), encoding="utf-8")
        self._send(200, b"ok", "text/plain")

    def log_message(self, fmt: str, *args: Any) -> None:
        message = "[istanbul-listener] " + fmt % args
        try:
            if self.log_file is not None:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(message + "\n")
            else:
                print(message, flush=True)
        except Exception:
            # Logging must never break coverage collection.
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6972)
    parser.add_argument("--out-dir", default="analysis/webrled_code_coverage/istanbul_listener")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    CoverageHandler.out_dir = Path(args.out_dir)
    CoverageHandler.log_file = Path(args.log_file) if args.log_file else None
    server = ThreadingHTTPServer((args.host, args.port), CoverageHandler)
    startup = f"[istanbul-listener] serving http://{args.host}:{args.port}/coverage"
    if CoverageHandler.log_file is not None:
        CoverageHandler.log_file.parent.mkdir(parents=True, exist_ok=True)
        CoverageHandler.log_file.write_text(startup + "\n", encoding="utf-8")
    else:
        print(startup, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
