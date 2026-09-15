#!/usr/bin/env python3
"""Subgroup analysis script — breaks down validation accuracy by demographic groups.

Usage:
    python scripts/subgroup_analysis.py \\
        --results /data/validation_results.csv \\
        --demographics /data/demographics.csv \\
        --output-dir /data/subgroup_reports/

The demographics CSV must have columns:
    patient_id, age, sex, bmi, race, scanner_vendor

Outputs per subgroup:
    subgroup_report.csv — accuracy by subgroup
    subgroup_report.html — visual dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("subgroup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_results(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_demographics(path: Path) -> Dict[str, Dict[str, str]]:
    """Returns dict: patient_id -> demographic fields."""
    out: Dict[str, Dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            pid = row.get("patient_id", "")
            if pid:
                out[pid] = row
    return out


def compute_subgroup_metrics(
    rows: List[Dict[str, Any]],
    group_field: str,
    ef_ai_field: str = "ef_ai",
    ef_gt_field: str = "ef_cardiologist",
) -> List[Dict[str, Any]]:
    """Compute EF MAE for each value of group_field."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            ef_ai = float(r[ef_ai_field])
            ef_gt = float(r[ef_gt_field])
            group = str(r.get(group_field, "unknown"))
            groups[group].append(abs(ef_ai - ef_gt))
        except (ValueError, TypeError, KeyError):
            continue

    out: List[Dict[str, Any]] = []
    for group, errors in sorted(groups.items()):
        if not errors:
            continue
        out.append({
            "subgroup_field": group_field,
            "subgroup_value": group,
            "n": len(errors),
            "mae": round(sum(errors) / len(errors), 2),
            "max_error": round(max(errors), 2),
            "min_error": round(min(errors), 2),
        })
    return out


def age_to_decade(age_str: str) -> str:
    try:
        age = int(float(age_str))
        return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
    except (ValueError, TypeError):
        return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Subgroup analysis for cardio-echo-suite validation")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--demographics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./subgroup_reports"))
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    demos = load_demographics(args.demographics)

    # Merge demographics into results
    for r in results:
        pid = r.get("patient_id", "")
        demo = demos.get(pid, {})
        r["age_decade"] = age_to_decade(demo.get("age", ""))
        r["sex"] = demo.get("sex", "unknown")
        r["bmi_category"] = _bmi_category(demo.get("bmi", ""))
        r["race"] = demo.get("race", "unknown")
        r["scanner_vendor"] = demo.get("scanner_vendor", "unknown")

    # Compute subgroup metrics
    all_subgroups: List[Dict[str, Any]] = []
    for field in ["age_decade", "sex", "bmi_category", "race", "scanner_vendor"]:
        all_subgroups.extend(compute_subgroup_metrics(results, field))

    # Write CSV
    csv_path = args.output_dir / "subgroup_report.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subgroup_field", "subgroup_value", "n", "mae", "max_error", "min_error"])
        writer.writeheader()
        for row in all_subgroups:
            writer.writerow(row)
    logger.info("Wrote %s", csv_path)

    # Write HTML dashboard
    html_path = args.output_dir / "subgroup_report.html"
    html_path.write_text(_generate_html(all_subgroups))
    logger.info("Wrote %s", html_path)

    # Flag equity gaps (any subgroup >10% above overall MAE)
    overall_mae = sum(r["mae"] * r["n"] for r in all_subgroups) / max(sum(r["n"] for r in all_subgroups), 1)
    equity_gaps = [r for r in all_subgroups if r["mae"] > overall_mae * 1.10]
    if equity_gaps:
        logger.warning("EQUITY GAPS DETECTED (MAE > 10%% above overall of %.2f):", overall_mae)
        for r in equity_gaps:
            logger.warning("  %s=%s: MAE=%.2f (n=%d)", r["subgroup_field"], r["subgroup_value"], r["mae"], r["n"])
        gap_path = args.output_dir / "equity_gaps.json"
        gap_path.write_text(json.dumps(equity_gaps, indent=2))
        logger.info("Wrote %s", gap_path)

    return 0


def _bmi_category(bmi_str: str) -> str:
    try:
        bmi = float(bmi_str)
        if bmi < 18.5:
            return "underweight"
        if bmi < 25:
            return "normal"
        if bmi < 30:
            return "overweight"
        if bmi < 35:
            return "obese_I"
        if bmi < 40:
            return "obese_II"
        return "obese_III"
    except (ValueError, TypeError):
        return "unknown"


def _generate_html(subgroups: List[Dict[str, Any]]) -> str:
    rows_html = ""
    for s in subgroups:
        rows_html += f"""
        <tr>
          <td>{s['subgroup_field']}</td>
          <td>{s['subgroup_value']}</td>
          <td>{s['n']}</td>
          <td>{s['mae']:.2f}</td>
          <td>{s['max_error']:.2f}</td>
          <td>{s['min_error']:.2f}</td>
        </tr>"""
    return f"""<!DOCTYPE html>
<html><head><title>Subgroup Analysis — cardio-echo-suite</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background: #1a3a5c; color: white; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
h1 {{ color: #1a3a5c; }}
</style></head><body>
<h1>Subgroup Analysis — cardio-echo-suite validation</h1>
<table>
  <tr><th>Subgroup field</th><th>Value</th><th>N</th><th>MAE (EF%)</th><th>Max error</th><th>Min error</th></tr>
  {rows_html}
</table>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
