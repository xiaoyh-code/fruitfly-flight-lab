"""Fetch only the official Crazyflie model subtree, with provenance and SHA256."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7'
PREFIX = 'bitcraze_crazyflie_2/'
REPO = 'google-deepmind/mujoco_menagerie'

def read(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'FruitFly-local-setup'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()

def main():
    tree = json.loads(read(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    entries = [item for item in tree['tree'] if item['type'] == 'blob' and item['path'].startswith(PREFIX)]
    def download(item):
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item["path"]}'
        target = ROOT / 'models' / item['path']
        data = read(url)
        if len(data) != item['size']:
            raise RuntimeError(f'Size mismatch: {target}')
        blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob_sha != item['sha']:
            raise RuntimeError(f'Git blob checksum mismatch: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {'path': str(target.relative_to(ROOT)), 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha': blob_sha}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download, entries))
    manifest = {'upstream': f'https://github.com/{REPO}', 'commit': COMMIT, 'files': files, 'total_bytes': sum(item['bytes'] for item in files)}
    (ROOT/'models'/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Downloaded and verified {len(files)} files, {manifest["total_bytes"]:,} bytes.')

if __name__ == '__main__':
    main()
