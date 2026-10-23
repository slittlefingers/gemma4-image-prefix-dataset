"""Verify and extract the fixed-prefix release without changing earlier rounds."""
import hashlib,json,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
EXPECTED={
 'gemma4-fixed991-1000-metadata-target.tar.gz':'4e7f476d13eba31ea8628054a79d6e095b02fca5ea121f107d8925286f8c7260',
 'gemma4-fixed991-1000-train-01.tar.gz':'c082bd5033bde382a3db4ad5c443568db314093c3a38aab79277171a44c04a76',
 'gemma4-fixed991-1000-train-02.tar.gz':'db5148f162b8e6a0eb7f830313ce1ef41d5983592721ed0c4a58a8b6811be7ee',
}

def main():
    audit=[]
    for name,expected in EXPECTED.items():
        p=REPO/'downloads/20261023_fixed991'/name
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,(name,actual)
        with tarfile.open(p) as t:
            members=t.getmembers()
            assert all(m.name.startswith('data/20261023_fixed991_1000/') for m in members)
            t.extractall(REPO,filter='data')
        audit.append(dict(file=name,bytes=p.stat().st_size,sha256=actual,members=len(members)))
        print('EXTRACTED',json.dumps(audit[-1]),flush=True)
    (HERE/'download_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
