"""Retrieve exactly the 56 dataset JPEGs, checking recorded content hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'data/20261022_pilot_prefill'


def download(item):
    path=HERE/'images'/f"{item['id']}.jpg"
    if not path.exists():
        with urllib.request.urlopen(item['source_url'],timeout=60) as response:
            content=response.read()
        assert hashlib.sha256(content).hexdigest()==item['source_sha256'],item['id']
        path.write_bytes(content)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==item['source_sha256'],item['id']
    return dict(id=item['id'],split=item['split'],sha256=actual,source=item['source_url'])


if __name__=='__main__':
    (HERE/'images').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'image_manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:
        audit=list(pool.map(download,manifest))
    (HERE/'download_audit.json').write_text(json.dumps(audit,indent=2))
    print('Downloaded and verified original JPEGs:',len(audit),flush=True)
