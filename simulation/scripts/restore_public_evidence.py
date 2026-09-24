#!/usr/bin/env python3
"""Restore hash-verified published action heads and replay; never overwrite different files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = 'hall-20260922-replay1'
MODELS = {
    'source-baseline.pt': ('checkpoints/missions/hall-20260921-fast/hall-visual-head.pt',
                         'b2134685720b09b1e7b93103ec639d1bb94daa0701bfeb9656655bb0fc8f76fa'),
    'selected-round-0.pt': (f'checkpoints/missions/{RUN}/hall-visual-head.pt',
                          '9d19c748edd43e9c517c9f10146be819446322b01ae7b0bb61b81da4c5ae609b'),
}


def restore(source, target, expected):
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'Published asset hash mismatch: {source}')
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f'Refusing to overwrite a different local file: {target}')
        print(f'Already verified: {target.relative_to(ROOT)}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a checkpoint created concurrently.
    with target.open('xb') as stream:
        stream.write(payload)
    print(f'Restored: {target.relative_to(ROOT)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT.parent / 'site', help='Published site directory beside simulation/')
    args = parser.parse_args()
    site = args.site.resolve()
    if not (site / 'publication-manifest.json').is_file():
        parser.error('Missing sibling site/. Clone the complete repository, or pass --site PATH.')
    manifest = json.loads((site / 'publication-manifest.json').read_text(encoding='utf-8'))
    hashes = {item['path']: item['sha256'] for item in manifest['files']}
    # Validate every requested input before restoring any file.
    items = [(site / f'research/{RUN}/models/{name}', ROOT / dest, sha)
             for name, (dest, sha) in MODELS.items()]
    replay = 'data/hall-replay.json'
    items.append((site / replay, ROOT / 'outputs/scene-replay/hall-replay.json', hashes[replay]))
    for source, target, digest in items:
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or modified published input: {source}')
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Refusing to overwrite a different local file: {target}')
    for source, target, digest in items:
        restore(source, target, digest)
    print('Restored published weights/replay, not a new training result. Original tensor values are unchanged.')


if __name__ == '__main__':
    main()
