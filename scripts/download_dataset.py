#!/usr/bin/env python3
"""
Download LUNA dataset from Zenodo.

This script downloads the LUNA25 dataset files from Zenodo using the API.
Requires ZENODO_TOKEN in .env file or as environment variable.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output /custom/path
    python scripts/download_dataset.py --record-id 14223624
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def get_zenodo_token() -> str:
    """Get Zenodo token from environment or .env file."""
    # First check environment variable
    token = os.environ.get('ZENODO_TOKEN')
    if token:
        return token

    # Then check .env file in project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    env_vars = load_env_file(env_file)

    token = env_vars.get('ZENODO_TOKEN')
    if not token:
        raise ValueError(
            "ZENODO_TOKEN not found. Please set it in .env file or as environment variable."
        )
    return token


def get_record_metadata(record_id: str, access_token: str) -> dict:
    """Fetch record metadata from Zenodo API."""
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, params={'access_token': access_token})

    if response.status_code == 401:
        raise PermissionError("Invalid Zenodo access token.")
    elif response.status_code == 404:
        raise ValueError(f"Record {record_id} not found on Zenodo.")
    elif response.status_code != 200:
        raise RuntimeError(f"Error retrieving record: {response.status_code} - {response.text}")

    return response.json()


def download_file(
    url: str,
    output_path: Path,
    access_token: str,
    expected_size: Optional[int] = None,
) -> None:
    """Download a single file with progress bar."""
    # Check if file already exists and is complete
    if output_path.exists():
        if expected_size and output_path.stat().st_size == expected_size:
            print(f"  Skipping (already exists): {output_path.name}")
            return
        else:
            print(f"  File exists but incomplete, re-downloading: {output_path.name}")

    # Download with streaming
    with requests.get(url, params={'access_token': access_token}, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0)) or expected_size

        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {output_path.name}",
                ncols=80,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def download_dataset(
    record_id: str,
    output_folder: Path,
    access_token: str,
    file_filter: Optional[str] = None,
) -> None:
    """Download all files from a Zenodo record."""
    print(f"Fetching record metadata for {record_id}...")
    metadata = get_record_metadata(record_id, access_token)

    files = metadata.get('files', [])
    if not files:
        print("No files found in this record.")
        return

    # Apply file filter if specified
    if file_filter:
        files = [f for f in files if file_filter.lower() in f['key'].lower()]
        if not files:
            print(f"No files matching filter '{file_filter}' found.")
            return

    # Calculate total size
    total_size = sum(f.get('size', 0) for f in files)
    total_size_gb = total_size / (1024 ** 3)

    print(f"\nRecord: {metadata.get('metadata', {}).get('title', 'Unknown')}")
    print(f"Files to download: {len(files)}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Output folder: {output_folder}")
    print()

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Download each file
    for index, file_info in enumerate(files, 1):
        filename = file_info['key']
        url = file_info['links']['self']
        expected_size = file_info.get('size')
        output_path = output_folder / filename

        print(f"[{index}/{len(files)}] Downloading: {filename}")

        try:
            download_file(url, output_path, access_token, expected_size)
        except requests.exceptions.RequestException as e:
            print(f"  Error downloading {filename}: {e}")
            continue

    print("\nAll downloads completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Download LUNA dataset from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output ../LUNA
    python scripts/download_dataset.py --filter subset0
        """,
    )
    parser.add_argument(
        '--record-id',
        default='14223624',
        help='Zenodo record ID (default: 14223624 for LUNA25)',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output folder for downloaded files (default: data/LUNA inside project)',
    )
    parser.add_argument(
        '--filter', '-f',
        type=str,
        default=None,
        help='Filter files by name (e.g., "subset0" to only download subset0)',
    )

    args = parser.parse_args()

    # Set default output path (inside project directory)
    if args.output is None:
        project_root = Path(__file__).parent.parent
        args.output = project_root / 'data' / 'LUNA'

    try:
        access_token = get_zenodo_token()
        download_dataset(
            record_id=args.record_id,
            output_folder=args.output,
            access_token=access_token,
            file_filter=args.filter,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Authentication error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(130)


if __name__ == '__main__':
    main()
