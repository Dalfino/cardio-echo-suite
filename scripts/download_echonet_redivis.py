#!/usr/bin/env python3
"""Selective EchoNet-Dynamic downloader via Redivis API.

Downloads only what you need (e.g., test split = 2,437 videos, ~1.5 GB)
instead of the full 10,030 videos (~7 GB).

Prerequisites:
  - Redivis account (free, register at https://redivis.com)
  - Redivis API token (Account Settings → API Tokens)

Usage:
  # Download only the test split (recommended for research shakedown)
  python scripts/download_echonet_redivis.py \\
      --token YOUR_REDIVIS_TOKEN \\
      --split TEST \\
      --dest /data/public/echonet-dynamic

  # Download only first 100 videos (smoke test)
  python scripts/download_echonet_redivis.py \\
      --token YOUR_REDIVIS_TOKEN \\
      --max-n 100 \\
      --dest /data/public/echonet-dynamic

  # Download all splits (will be ~7 GB total)
  python scripts/download_echonet_redivis.py \\
      --token YOUR_REDIVIS_TOKEN \\
      --split ALL \\
      --dest /data/public/echonet-dynamic

After download, run shakedown:
  python scripts/public_data_shakedown.py \\
      --dataset echonet-dynamic \\
      --data-dir /data/public/echonet-dynamic/Videos \\
      --labels /data/public/echonet-dynamic/FileList.csv \\
      --orchestrator-url http://localhost:8080 \\
      --output /tmp/shakedown.json \\
      --max-n 50 \\
      --i-understand-this-is-not-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import urllib.request
import urllib.error

logger = logging.getLogger("echonet-downloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Redivis API base
REDIVIS_API = "https://redivis.com/api/v1"
ECHONET_DATASET = "stanford-echonet/dynamic-1k0hr"  # public EchoNet-Dynamic on Redivis


def api_get(url: str, token: str, timeout: int = 60) -> Any:
    """Make an authenticated GET request to Redivis API."""
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.error("HTTP %d on %s", e.code, url)
        if e.code == 401:
            logger.error("Authentication failed. Check your Redivis API token.")
        elif e.code == 404:
            logger.error("Dataset not found. Check the dataset path.")
        try:
            err_body = e.read().decode("utf-8")
            logger.error("Response: %s", err_body[:500])
        except Exception:
            pass
        raise
    except urllib.error.URLError as e:
        logger.error("URL error: %s", e.reason)
        raise


def download_file(url: str, token: str, dest: Path, timeout: int = 120) -> bool:
    """Download a single file. Returns True on success."""
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)  # 64 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def list_dataset_files(token: str, dataset: str = ECHONET_DATASET) -> List[Dict[str, Any]]:
    """List all files in the EchoNet-Dynamic Redivis dataset."""
    # Try the listing endpoint
    url = f"{REDIVIS_API}/datasets/{dataset}/files?perPage=1000"
    logger.info("Listing files in %s ...", dataset)
    files: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        page_url = url
        if page_token:
            page_url += f"&pageToken={page_token}"
        data = api_get(page_url, token)
        page_files = data.get("results", data.get("files", []))
        files.extend(page_files)
        logger.info("  Listed %d files so far...", len(files))

        page_token = data.get("nextPageToken")
        if not page_token or not page_files:
            break

    logger.info("Total files found: %d", len(files))
    return files


def download_filelist_csv(token: str, dataset: str, dest: Path) -> bool:
    """Download the FileList.csv (labels)."""
    # Try common locations
    possible_paths = ["FileList.csv", "data/FileList.csv", "FileList.csv.gz"]
    for path in possible_paths:
        url = f"{REDIVIS_API}/datasets/{dataset}/files/{quote(path)}/content"
        logger.info("Trying FileList.csv at: %s", path)
        if download_file(url, token, dest / "FileList.csv"):
            logger.info("✅ Downloaded FileList.csv (%d bytes)", (dest / "FileList.csv").stat().st_size)
            return True
    logger.error("Could not find FileList.csv in dataset")
    return False


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Selective EchoNet-Dynamic downloader via Redivis API")
    p.add_argument("--token", required=True, help="Redivis API token (from Account Settings → API Tokens)")
    p.add_argument("--dataset", default=ECHONET_DATASET, help="Redivis dataset path")
    p.add_argument("--dest", type=Path, default=Path("/data/public/echonet-dynamic"), help="Destination directory")
    p.add_argument("--split", choices=["TRAIN", "VAL", "TEST", "ALL"], default="TEST",
                   help="Which split to download (default: TEST = 2,437 videos)")
    p.add_argument("--max-n", type=int, default=0, help="Max number of videos to download (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = p.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)

    # Step 1: Try to download FileList.csv (labels)
    logger.info("=" * 60)
    logger.info("Step 1: Downloading FileList.csv (labels)")
    logger.info("=" * 60)
    filelist_path = args.dest / "FileList.csv"
    if filelist_path.exists():
        logger.info("FileList.csv already exists, skipping")
    else:
        download_filelist_csv(args.token, args.dataset, args.dest)

    # Parse FileList.csv to figure out which videos belong to which split
    import csv
    if not filelist_path.exists():
        logger.error("Cannot proceed without FileList.csv")
        return 1

    videos_to_download: List[str] = []  # filenames like "0X10A28877E97DF540.avi"
    with filelist_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("Split", "").upper()
            filename = row.get("FileName", "")
            if not filename:
                continue
            if not filename.endswith(".avi"):
                filename += ".avi"
            if args.split == "ALL" or split == args.split:
                videos_to_download.append(filename)

    if args.max_n > 0:
        videos_to_download = videos_to_download[:args.max_n]

    logger.info("Will download %d videos (split=%s)", len(videos_to_download), args.split)

    if args.dry_run:
        for v in videos_to_download[:10]:
            logger.info("  would download: %s", v)
        if len(videos_to_download) > 10:
            logger.info("  ... and %d more", len(videos_to_download) - 10)
        return 0

    # Step 2: Download videos
    logger.info("=" * 60)
    logger.info("Step 2: Downloading %d videos", len(videos_to_download))
    logger.info("=" * 60)

    videos_dir = args.dest / "Videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    t0 = time.perf_counter()

    for i, filename in enumerate(videos_to_download, 1):
        dest_path = videos_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            success_count += 1
            if i % 50 == 0:
                logger.info("[%d/%d] %s already exists, skipping", i, len(videos_to_download), filename)
            continue

        # Try multiple possible paths in the dataset
        possible_paths = [
            f"Videos/{filename}",
            f"videos/{filename}",
            filename,
            f"data/Videos/{filename}",
        ]

        downloaded = False
        for path in possible_paths:
            url = f"{REDIVIS_API}/datasets/{args.dataset}/files/{quote(path)}/content"
            if download_file(url, args.token, dest_path):
                downloaded = True
                break

        if downloaded:
            success_count += 1
        else:
            fail_count += 1

        if i % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (len(videos_to_download) - i) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] %s | success=%d fail=%d | %.1f videos/s | ETA: %.0fs",
                i, len(videos_to_download), filename, success_count, fail_count,
                rate, eta
            )

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("✅ Done in %.1f seconds", elapsed)
    logger.info("   Success: %d / %d", success_count, len(videos_to_download))
    logger.info("   Failed:  %d / %d", fail_count, len(videos_to_download))
    logger.info("   Location: %s", videos_dir)
    logger.info("=" * 60)

    if fail_count > 0:
        logger.warning("%d downloads failed. Try re-running the script — it will skip already-downloaded files.", fail_count)
        return 1

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify: ls %s | wc -l  (should be %d)", videos_dir, success_count)
    logger.info("  2. Run shakedown:")
    logger.info("     python scripts/public_data_shakedown.py \\")
    logger.info("         --dataset echonet-dynamic \\")
    logger.info("         --data-dir %s \\")
    logger.info("         --labels %s \\")
    logger.info("         --orchestrator-url http://localhost:8080 \\")
    logger.info("         --output /tmp/echo_shakedown.json \\")
    logger.info("         --max-n 50 \\")
    logger.info("         --i-understand-this-is-not-validation", videos_dir, filelist_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
