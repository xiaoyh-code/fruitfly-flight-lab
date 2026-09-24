#!/usr/bin/env python3
"""Download the original Hall scene directly from its author and verify it.

The PLY is not redistributed by this project. Upstream marks it all rights
reserved; see models/gaussian/LICENSE and SOURCE-README.md. A verified existing
file is reused. Failed downloads never replace the destination file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
# Pinned to the locally recorded upstream artifact in provenance.json.
URL = "https://huggingface.co/datasets/amacati/splats/resolve/main/robot_hall.ply"
EXPECTED_BYTES = 6_689_711
EXPECTED_SHA256 = "ac883665efc87f84174f0a7d385277299da39e49df7c3838d78dad860ddf2681"
DESTINATION = ROOT / "models/gaussian/robot_hall.ply"


def verified(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != EXPECTED_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == EXPECTED_SHA256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify the local PLY; do not use the network")
    parser.add_argument("--force", action="store_true", help="Replace an invalid local PLY, only after a verified download")
    args = parser.parse_args()
    if verified(DESTINATION):
        print(f"PASS: Hall scene verified ({EXPECTED_BYTES:,} bytes): {DESTINATION}")
        return 0
    if args.check_only:
        print("FAIL: Hall scene is missing or its size/SHA256 differs.\n"
              "Run: python scripts/download_hall_scene.py", file=sys.stderr)
        return 1
    if DESTINATION.exists() and not args.force:
        print("Existing Hall scene does not match the recorded upstream artifact. "
              "It has been preserved. Use --force to replace it with a verified download.", file=sys.stderr)
        return 1

    print(f"Downloading {EXPECTED_BYTES:,} bytes (6.69 MB) from the scene author.\n{URL}")
    print("Upstream licence: all rights reserved. This script does not grant redistribution rights.")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(URL, headers={"User-Agent": "FruitFlyFlightLab/1.0"})
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".robot_hall-", suffix=".part",
                                         dir=DESTINATION.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(request, timeout=90) as response:
                received = 0
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > EXPECTED_BYTES:
                        raise ValueError("Download exceeds the pinned artifact size; upstream may have changed")
                    output.write(block)
        if not verified(temporary):
            raise ValueError("Size or SHA256 mismatch; refusing to install this download")
        os.replace(temporary, DESTINATION)
        print(f"PASS: verified SHA256 {EXPECTED_SHA256}\nSaved: {DESTINATION}")
        return 0
    except (OSError, URLError, ValueError) as error:
        print(f"FAIL: {error}\nDestination left unchanged. Check the connection and upstream availability.", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
