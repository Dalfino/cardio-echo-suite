"""Tests for the EchoPrime fine-tuning scaffold (no GPU required)."""
import json
import tempfile
from pathlib import Path

from app.finetune import EchoReportDataset, EchoReportExample, build_lora_config


def _make_jsonl(tmp_path: Path, n: int = 3) -> Path:
    rows = []
    for i in range(n):
        rows.append({
            "video_path": f"/data/echo/study{i}.mp4",
            "view": "A4C" if i % 2 == 0 else "PLAX",
            "ef": 55.0 + i,
            "report_text": f"Normal study {i}. EF = {55.0+i}%.",
        })
    p = tmp_path / "reports.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_dataset_loads_examples():
    with tempfile.TemporaryDirectory() as td:
        p = _make_jsonl(Path(td), n=3)
        ds = EchoReportDataset(p)
        assert len(ds) == 3
        ex = ds[0]
        assert ex["view"] == "A4C"
        assert ex["ef"] == 55.0
        assert "EF" in ex["report_text"]


def test_example_extra_fields():
    ex = EchoReportExample.from_dict({
        "video_path": "/x.mp4",
        "view": "A2C",
        "ef": 60.0,
        "report_text": "ok",
        "patient_id": "P1",
        "extra_field": "value",
    })
    assert ex.extra["patient_id"] == "P1"
    assert ex.extra["extra_field"] == "value"


def test_lora_config_built():
    cfg = build_lora_config(r=8, alpha=16)
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert "q_proj" in cfg.target_modules
