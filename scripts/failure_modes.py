#!/usr/bin/env python3
"""Failure mode dashboard generator — picks the worst N cases for manual review.

Usage:
    python scripts/failure_modes.py \\
        --results /data/validation_results.csv \\
        --output /data/failure_modes.html \\
        --n 20

Produces an HTML dashboard with the worst N cases (by abs error) for
cardiologist review. Includes patient_id, study_uid, AI EF, cardiologist EF,
and abs error, sorted descending by error.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("failure_modes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_results(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def generate_dashboard(results: List[Dict[str, str]], n: int = 20) -> str:
    """Generate HTML dashboard of worst N cases."""
    # Filter rows with valid abs_error
    valid: List[Dict[str, Any]] = []
    for r in results:
        try:
            err = float(r.get("abs_error", ""))
            if err >= 0:
                valid.append({**r, "abs_error_float": err})
        except (ValueError, TypeError):
            continue

    # Sort by abs_error descending
    valid.sort(key=lambda r: r["abs_error_float"], reverse=True)
    worst = valid[:n]

    rows_html = ""
    for i, r in enumerate(worst, 1):
        rows_html += f"""
        <tr>
          <td>{i}</td>
          <td>{r.get('patient_id', '')}</td>
          <td>{r.get('study_uid', '')[-16:]}</td>
          <td>{r.get('ef_ai', '')}</td>
          <td>{r.get('ef_cardiologist', '')}</td>
          <td class="error">{r['abs_error_float']:.2f}</td>
          <td>{r.get('inference_s', '')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>Failure Mode Dashboard — cardio-echo-suite</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background: #1a3a5c; color: white; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
td.error {{ color: #c00; font-weight: bold; }}
h1 {{ color: #1a3a5c; }}
.summary {{ background: #fee; padding: 12px; border-radius: 4px; margin: 16px 0; }}
</style></head><body>
<h1>Failure Mode Dashboard — cardio-echo-suite validation</h1>
<div class="summary">
  <p><strong>Total studies:</strong> {len(results)}</p>
  <p><strong>Valid predictions:</strong> {len(valid)}</p>
  <p><strong>Worst {n} cases shown below</strong> — review these with the cardiologist to identify failure modes (e.g. poor image quality, arrhythmia, unusual anatomy).</p>
</div>
<table>
  <tr><th>Rank</th><th>Patient ID</th><th>Study UID (last 16 chars)</th><th>AI EF (%)</th><th>Cardiologist EF (%)</th><th>Abs error</th><th>Inference (s)</th></tr>
  {rows_html}
</table>
</body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate failure mode dashboard")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("failure_modes.html"))
    p.add_argument("--n", type=int, default=20, help="Number of worst cases to include")
    args = p.parse_args(argv)

    results = load_results(args.results)
    logger.info("Loaded %d results", len(results))

    html = generate_dashboard(results, n=args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    logger.info("Wrote %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
