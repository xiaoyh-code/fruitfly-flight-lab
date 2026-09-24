"""Download the official small pretrained ensemble and verify upstream SHA256."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'flyvis'
URL = 'https://drive.google.com/uc?export=download&id=13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC'
EXPECTED = '71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST/'results_pretrained_models.zip'
    request = urllib.request.Request(URL, headers={'User-Agent': 'FruitFly-local-setup'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED:
        raise RuntimeError(f'Official model checksum mismatch; not extracting: {digest}')
    archive.write_bytes(data)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (DEST/member.filename).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise RuntimeError('Unexpected archive path')
        zipped.extractall(DEST)
    (DEST/'provenance.json').write_text(json.dumps({'url':URL,'bytes':len(data),'sha256':digest,'checksum_source':'https://github.com/TuragaLab/flyvis/blob/main/flyvis_cli/download_pretrained_models.py'},indent=2)+'\n', encoding='utf-8')
    print(f'Official Flyvis weights verified and extracted: {len(data):,} bytes.')

if __name__ == '__main__':
    main()
