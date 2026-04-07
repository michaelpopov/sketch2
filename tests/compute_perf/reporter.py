#!/usr/bin/env python3
"""Aggregate compute perf runner logs into a compact table."""

from __future__ import annotations

import re
from pathlib import Path

from common import load_config


REPORT_DISTANCE_RE = re.compile(r"^Distance:\s*(\S+)\s*$")
REPORT_AVG_RE = re.compile(r"^Avg Time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$")


def log_dir(config) -> Path:
    return config.db_dir / "logs"


def runner_log_path(config, engine: str) -> Path:
    return log_dir(config) / f"runner_{engine}.log"


def runner_dist_log_path(config, engine: str, dist: str) -> Path:
    return log_dir(config) / f"runner_{engine}_{dist}.log"


def parse_runner_log(path: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    pending_dist: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            dist_match = REPORT_DISTANCE_RE.match(line)
            if dist_match:
                pending_dist = dist_match.group(1).lower()
                continue

            avg_match = REPORT_AVG_RE.match(line)
            if avg_match and pending_dist is not None:
                results[pending_dist] = float(avg_match.group(1))
                pending_dist = None

    return results


def format_cell(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}s"


def build_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(cells: list[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        return " | ".join(padded)

    divider = "-+-".join("-" * width for width in widths)
    parts = [render_row(headers), divider]
    for row in rows:
        parts.append(render_row(row))
    return "\n".join(parts)


def main() -> None:
    config = load_config()

    headers = ["engine"] + config.dist_funcs
    rows: list[list[str]] = []

    for engine in config.compute_engines:
        row = [engine]
        for dist in config.dist_funcs:
            dist_path = runner_dist_log_path(config, engine, dist)
            if dist_path.exists():
                metrics = parse_runner_log(dist_path)
                row.append(format_cell(metrics.get(dist)))
                continue

            path = runner_log_path(config, engine)
            metrics = parse_runner_log(path) if path.exists() else {}
            row.append(format_cell(metrics.get(dist)))
        rows.append(row)

    print("--- PERFORMANCE SUMMARY (avg time) ---")
    print(build_table(headers, rows))
    print("--------------------------------------")


if __name__ == "__main__":
    main()
