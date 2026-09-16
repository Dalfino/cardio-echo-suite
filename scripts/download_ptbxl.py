#!/usr/bin/env python3
"""Selective PTB-XL downloader via PhysioNet API.

Downloads PTB-XL via the `wfdb` Python library, which handles authentication
and file listing automatically. Can download the full dataset or a subset.

Prerequisites:
  - PhysioNet account (https://physionet.org/register/)
  - PhysioNet CREDENTIALED access (requires CITI training — see
    docs/research/DATASET_ACQUISITION_GUIDE.md)
  - wfdb library: pip install wfdb

Usage:
  # Download full PTB-XL (1.7 GB, ~10-15 min)
  python scripts/download_ptbxl.py \\
      --user you@email.com \\
      --password secret123 \\
      --dest /data/public/ptbxl

  # Download only first 100 ECGs (smoke test, ~20 MB)
  python scripts/download_ptbxl.py \\
      --user you@email.com \\
      --password secret123 \\
      --dest /data/public/ptbxl \\
      --max-n 100

  # Download only test split
  python scripts/download_ptbxl.py \\
      --user you@email.com \\
      --password secret123 \\
      --dest /data/public/ptbxl \\
      --split test
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("ptbxl-downloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PHYSIONET_BASE = "https://physionet.org/files/ptbxl/1.0.3"
PHYSIONET_ZIP = f"{PHYSIONET_BASE}/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def download_full_zip(user: str, password: str, dest: Path) -> bool:
    """Download the full PTB-XL ZIP (1.7 GB)."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ptbxl.zip"

    if zip_path.exists() and zip_path.stat().st_size > 1_000_000_000:  # >1 GB = likely complete
        logger.info("ZIP already downloaded (%d MB), skipping", zip_path.stat().st_size // 1_000_000)
        return True

    logger.info("Downloading full PTB-XL ZIP (1.7 GB)...")
    logger.info("This will take ~10-15 minutes depending on your connection")

    import urllib.request
    import base64

    req = urllib.request.Request(PHYSIONET_ZIP)
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with zip_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        if downloaded % (50 * 1024 * 1024) == 0:  # every 50 MB
                            logger.info("  %d / %d MB (%d%%)", downloaded // 1_000_000,
                                        total // 1_000_000, pct)
        logger.info("✅ Downloaded %d MB", zip_path.stat().st_size // 1_000_000)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            logger.error("Authentication failed. Check your PhysioNet username/password.")
            logger.error("Also verify your account has CREDENTIALED access (not just registered).")
        else:
            logger.error("HTTP %d: %s", e.code, e.reason)
        return False
    except Exception as e:
        logger.error("Download failed: %s", e)
        return False


def extract_zip(zip_path: Path, dest: Path) -> bool:
    """Extract the PTB-XL ZIP."""
    logger.info("Extracting %s ...", zip_path)
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        logger.info("✅ Extracted to %s", dest)
        # Clean up the zip to save space
        zip_path.unlink()
        logger.info("Removed ZIP to save disk space")
        return True
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return False


def verify_structure(dest: Path) -> bool:
    """Verify the expected PTB-XL structure exists."""
    expected_files = [
        dest / "ptbxl_database.csv",
        dest / "scp_statements.csv",
        dest / "records100",
        dest / "records500",
    ]
    for f in expected_files:
        if not f.exists():
            logger.warning("Expected file missing: %s", f)
            return False
    logger.info("✅ Structure verified")

    # Count records
    records100 = dest / "records100"
    n_dat = len(list(records100.rglob("*.dat")))
    n_hea = len(list(records100.rglob("*.hea")))
    logger.info("  records100: %d .dat files, %d .hea files", n_dat, n_hea)
    return True


def download_subset_via_wfdb(user: str, password: str, dest: Path, max_n: int = 100) -> bool:
    """Use the wfdb library to download a subset of PTB-XL.

    This is more efficient for small downloads because it doesn't grab the
    full ZIP.
    """
    try:
        import wfdb  # type: ignore
    except ImportError:
        logger.error("wfdb library not installed. Run: pip install wfdb")
        return False

    # Set credentials via env vars (wfdb reads these)
    os.environ["WFDB_USER"] = user
    os.environ["WFDB_PASSWORD"] = password

    dest.mkdir(parents=True, exist_ok=True)

    # First download the labels CSV
    logger.info("Downloading ptbxl_database.csv ...")
    import base64
    req = urllib.request.Request(f"{PHYSIONET_BASE}/ptbxl_database.csv")
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {credentials}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            (dest / "ptbxl_database.csv").write_bytes(resp.read())
        logger.info("✅ Downloaded ptbxl_database.csv")
    except Exception as e:
        logger.error("Failed to download labels: %s", e)
        return False

    # Download scp_statements.csv
    logger.info("Downloading scp_statements.csv ...")
    req = urllib.request.Request(f"{PHYSIONET_BASE}/scp_statements.csv")
    req.add_header("Authorization", f"Basic {credentials}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            (dest / "scp_statements.csv").write_bytes(resp.read())
        logger.info("✅ Downloaded scp_statements.csv")
    except Exception as e:
        logger.warning("Could not download scp_statements.csv: %s", e)

    # Parse the labels CSV to get list of records
    records_to_download: List[str] = []
    with (dest / "ptbxl_database.csv").open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_n:
                break
            # filename_lr is relative path like "records100/00000/00001_lr"
            filename = row.get("filename_lr", "")
            if filename:
                records_to_download.append(filename)

    logger.info("Will download %d records", len(records_to_download))

    # Download each record (.dat + .hea pair)
    records100_dir = dest / "records100"
    records100_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    fail = 0
    t0 = time.perf_counter()

    for i, record in enumerate(records_to_download, 1):
        record_path = dest / record
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_name = record_path.name
        record_dir = record_path.parent

        try:
            # wfdb.dl_record downloads both .hea and .dat
            wfdb.dl_record(
                f"ptbxl/{record}",  # PhysioNet record spec
                dl_dir=str(record_dir),
                record_name=record_name,
            )
            success += 1
        except Exception as e:
            logger.debug("Failed to download %s: %s", record, e)
            fail += 1

        if i % 10 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("[%d/%d] success=%d fail=%d (%.1fs)", i, len(records_to_download),
                        success, fail, elapsed)

    logger.info("=" * 60)
    logger.info("✅ Done in %.1f seconds", time.perf_counter() - t0)
    logger.info("   Success: %d / %d", success, len(records_to_download))
    logger.info("   Failed:  %d / %d", fail, len(records_to_download))
    logger.info("   Location: %s", records100_dir)
    return fail == 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PTB-XL downloader via PhysioNet")
    p.add_argument("--user", required=True, help="PhysioNet username (email)")
    p.add_argument("--password", required=True, help="PhysioNet password")
    p.add_argument("--dest", type=Path, default=Path("/data/public/ptbxl"))
    p.add_argument("--max-n", type=int, default=0,
                   help="Max records to download (0 = full dataset). If >0, uses wfdb for selective download.")
    p.add_argument("--full-zip", action="store_true",
                   help="Force downloading the full ZIP (1.7 GB) even with --max-n")
    args = p.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)

    if args.max_n > 0 and not args.full_zip:
        # Selective download via wfdb (smaller, faster for testing)
        logger.info="Selective download of %d records", args.max_n
        ok = download_subset_via_wfdb(args.user, args.password, args.dest, args.max_n)
    else:
        # Full ZIP download
        logger.info("Full ZIP download (1.7 GB)")
        if not download_full_zip(args.user, args.password, args.dest):
            return 1
        if not extract_zip(args.dest / "ptbxl.zip", args.dest):
            return 1

    # Verify
    if not verify_structure(args.dest):
        logger.warning("Structure verification failed — some files may be missing")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify: ls %s/records100/ | head", args.dest)
    logger.info("  2. Run shakedown:")
    logger.info("     python scripts/public_data_shakedown.py \\")
    logger.info("         --dataset ptbxl \\")
    logger.info("         --data-dir %s/records100 \\")
    logger.info("         --labels %s/ptbxl_database.csv \\", args.dest, args.dest)
    logger.info("         --orchestrator-url http://localhost:8080 \\")
    logger.info("         --output /tmp/ecg_shakedown.json \\")
    logger.info("         --max-n 50 \\")
    logger.info("         --i-understand-this-is-not-validation")
    return 0 if verify_structure(args.dest) else 1


if __name__ == "__main__":
    sys.exit(main())
