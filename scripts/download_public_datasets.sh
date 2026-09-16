#!/usr/bin/env bash
# download_public_datasets.sh — automated downloader for cardio-echo-suite research datasets
#
# Usage:
#   ./scripts/download_public_datasets.sh --dataset echonet
#   ./scripts/download_public_datasets.sh --dataset ptbxl --user YOUR_PHYSIONET_USER --pass YOUR_PHYSIONET_PASS
#   ./scripts/download_public_datasets.sh --dataset all --user YOUR_USER --pass YOUR_PASS --dest /data/public
#
# Prerequisites:
#   - For PTB-XL / MIMIC-IV-ECG: PhysioNet credentialed access (see docs/research/DATASET_ACQUISITION_GUIDE.md)
#   - For EchoNet-Dynamic: Stanford team approval (emailed separately)
#   - wget, unzip installed
#
# License reminder: All datasets are RESEARCH-ONLY. See DATASET_ACQUISITION_GUIDE.md for details.

set -euo pipefail

DEFAULT_DEST="/data/public"
ECHONET_URL=""  # Filled in after Stanford approval — they send a download link
PTBXL_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
MIMIC_ECG_URL="https://physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiographic-matched-subset-of-mimic-iv-1.1.zip"
ACDC_URL=""  # Requires registration at https://humanheart-project.creatis.insa-lyon.fr/database/

DATASET=""
DEST="$DEFAULT_DEST"
PHYSIO_USER=""
PHYSIO_PASS=""

print_usage() {
    cat <<EOF
Usage: $0 --dataset <name> [options]

Required:
  --dataset <name>     One of: echonet, ptbxl, mimic-ecg, acdc, all

Optional:
  --dest <path>        Destination directory (default: /data/public)
  --user <username>    PhysioNet username (required for ptbxl, mimic-ecg)
  --pass <password>    PhysioNet password (required for ptbxl, mimic-ecg)
  --echonet-url <url>  EchoNet-Dynamic download URL (from Stanford approval email)
  --help               Show this help

Examples:
  # Download EchoNet-Dynamic (after Stanford approval)
  $0 --dataset echonet --echonet-url "https://...link-from-stanford..."

  # Download PTB-XL (after PhysioNet credentialing)
  $0 --dataset ptbxl --user john@example.com --pass secret123

  # Download all to custom location
  $0 --dataset all --user john@example.com --pass secret123 --dest /media/external/data

Prerequisites:
  - PhysioNet credentialed access (https://physionet.org/register/)
  - For EchoNet-Dynamic: email echonet@stanford.edu for access
  - wget, unzip installed

⚠️  RESEARCH USE ONLY — see docs/research/DATASET_ACQUISITION_GUIDE.md
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --user) PHYSIO_USER="$2"; shift 2 ;;
        --pass) PHYSIO_PASS="$2"; shift 2 ;;
        --echonet-url) ECHONET_URL="$2"; shift 2 ;;
        --help|-h) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"
    print_usage
    exit 1
fi

# Check prerequisites
check_prereqs() {
    if ! command -v wget &> /dev/null; then
        echo "ERROR: wget is not installed. Install with: apt install wget"
        exit 1
    fi
    if ! command -v unzip &> /dev/null; then
        echo "ERROR: unzip is not installed. Install with: apt install unzip"
        exit 1
    fi
}

check_physio_creds() {
    if [[ -z "$PHYSIO_USER" || -z "$PHYSIO_PASS" ]]; then
        echo "ERROR: PhysioNet credentials required for this dataset."
        echo "Register at https://physionet.org/register/ and pass --user and --pass"
        exit 1
    fi
}

print_warning() {
    cat <<EOF

==============================================================
⚠️  RESEARCH USE ONLY  ⚠️
==============================================================
These datasets are licensed for RESEARCH purposes only.
- Cannot be used for commercial products
- Cannot be used to make clinical decisions on real patients
- Cannot be redistributed
- Must cite original papers in any publication

See docs/research/DATASET_ACQUISITION_GUIDE.md for details.
==============================================================

EOF
    sleep 3
}

download_echonet() {
    if [[ -z "$ECHONET_URL" ]]; then
        echo "ERROR: EchoNet-Dynamic requires manual approval from Stanford."
        echo "1. Email echonet@stanford.edu with your research purpose + CITI certificate"
        echo "2. They will respond with a download URL"
        echo "3. Re-run: $0 --dataset echonet --echonet-url \"https://...\""
        exit 1
    fi
    local dest="$DEST/echonet-dynamic"
    mkdir -p "$dest"
    echo "Downloading EchoNet-Dynamic to $dest..."
    wget -c "$ECHONET_URL" -O "$dest/EchoNet-Dynamic.zip"
    echo "Extracting..."
    unzip -q "$dest/EchoNet-Dynamic.zip" -d "$dest"
    rm "$dest/EchoNet-Dynamic.zip"
    echo "✅ EchoNet-Dynamic ready at $dest"
    echo "   Videos: $(ls $dest/Videos/ | wc -l) (expected: 10030)"
    echo "   Labels: $dest/FileList.csv"
}

download_ptbxl() {
    check_physio_creds
    local dest="$DEST/ptbxl"
    mkdir -p "$dest"
    echo "Downloading PTB-XL to $dest..."
    wget -c --user="$PHYSIO_USER" --password="$PHYSIO_PASS" \
        "$PTBXL_URL" -O "$dest/ptbxl.zip"
    echo "Extracting..."
    unzip -q "$dest/ptbxl.zip" -d "$dest"
    rm "$dest/ptbxl.zip"
    echo "✅ PTB-XL ready at $dest"
    echo "   Records: $(find $dest/records100 -name '*.dat' | wc -l)"
    echo "   Labels: $dest/ptbxl_database.csv"
}

download_mimic_ecg() {
    check_physio_creds
    local dest="$DEST/mimic-ecg"
    mkdir -p "$dest"
    echo "Downloading MIMIC-IV-ECG to $dest..."
    echo "(This is ~50 GB — may take several hours)"
    wget -c --user="$PHYSIO_USER" --password="$PHYSIO_PASS" \
        "$MIMIC_ECG_URL" -O "$dest/mimic-ecg.zip"
    echo "Extracting..."
    unzip -q "$dest/mimic-ecg.zip" -d "$dest"
    rm "$dest/mimic-ecg.zip"
    echo "✅ MIMIC-IV-ECG ready at $dest"
}

download_acdc() {
    local dest="$DEST/acdc"
    mkdir -p "$dest"
    echo "ACDC requires manual registration at:"
    echo "  https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00bc"
    echo "1. Register (free, instant)"
    echo "2. Download the ACDC dataset ZIP"
    echo "3. Move it to $dest/ and unzip"
    echo ""
    echo "After manual download, structure should be:"
    echo "  $dest/training/patient001/patient001_frame01.nii.gz"
    echo "  $dest/training/patient001/patient001_frame01_gt.nii.gz"
    echo "  ..."
}

# Main
check_prereqs
print_warning
mkdir -p "$DEST"

case "$DATASET" in
    echonet) download_echonet ;;
    ptbxl) download_ptbxl ;;
    mimic-ecg) download_mimic_ecg ;;
    acdc) download_acdc ;;
    all)
        echo "Downloading all datasets to $DEST..."
        download_echonet
        download_ptbxl
        download_mimic_ecg
        download_acdc
        ;;
    *) echo "ERROR: Unknown dataset '$DATASET'"; print_usage; exit 1 ;;
esac

echo ""
echo "============================================================"
echo "Next steps:"
echo "  1. Verify datasets: see docs/research/DATASET_ACQUISITION_GUIDE.md"
echo "  2. Run shakedown: python scripts/public_data_shakedown.py --help"
echo "  3. Start your paper: see docs/research/PAPER_DRAFT_TEMPLATE.md"
echo "============================================================"
