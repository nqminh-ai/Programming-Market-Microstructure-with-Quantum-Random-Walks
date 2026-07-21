"""Helper script to automatically download sample processed market data.

This allows new users to run the platform in live/calculation mode (DEMO_MODE=false)
without needing to download and clean gigabytes of raw tick data.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.paths import asset_data_dir

# Default sample data URL (e.g. hosted on GitHub Releases or S3)
DEFAULT_URL = (
    "https://github.com/nqminh-ai/"
    "Programming-Market-Microstructure-with-Quantum-Random-Walks/"
    "releases/download/v1.0.0-sample/sample_processed_data.zip"
)


def download_and_extract_sample(url: str, target_symbol: str) -> bool:
    """Return True on success, False on any download/extraction failure."""
    print(f"[*] Downloading sample data from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to download sample data: {e}")
        print("[TIP] Make sure the URL is accessible or provide a custom URL using --url parameter.")
        return False

    # Extract target directories using src.data.paths
    processed_dir = asset_data_dir(target_symbol, "processed", create=True)
    features_dir = asset_data_dir(target_symbol, "features", create=True)

    print("[*] Extracting zip contents to target data directories...")
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Check the zip structure and extract files
            for file_info in zip_ref.infolist():
                filename = Path(file_info.filename).name
                if not filename:
                    continue  # skip directory entries

                # Determine target kind based on folder name in zip or file prefix
                if "processed" in file_info.filename:
                    dest_path = processed_dir / filename
                elif "features" in file_info.filename:
                    dest_path = features_dir / filename
                else:
                    # Default fallback based on naming pattern
                    if "processed" in filename:
                        dest_path = processed_dir / filename
                    else:
                        dest_path = features_dir / filename

                with zip_ref.open(file_info) as source, open(dest_path, "wb") as target:
                    target.write(source.read())
                print(f"   [OK] Extracted: {dest_path.relative_to(ROOT)}")
                
        print("\n[PASS] Sample data successfully downloaded and aligned to paths!")
        print(f"   - Processed Ticks: {processed_dir.relative_to(ROOT)}")
        print(f"   - Features:        {features_dir.relative_to(ROOT)}")
        print("\nYou can now run the app in calculation mode: DEMO_MODE=false")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to extract zip archive: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample market data for QRW platform")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="URL of the zipped sample data",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Target symbol directory (default: BTCUSDT)",
    )
    args = parser.parse_args()

    if not download_and_extract_sample(args.url, args.symbol):
        sys.exit(1)


if __name__ == "__main__":
    main()
