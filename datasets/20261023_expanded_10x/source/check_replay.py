"""Fresh real-phone replay with changed prefill boundaries at long prefixes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump, EXPERIMENTS
from kvrecon.npu import NpuBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data',type=Path)
    parser.add_argument('--preserve-blocks',action='store_true')
    args = parser.parse_args()
    root = args.data
    protocol = json.loads((root/'protocol.json').read_text())
    index = json.loads((root/'index.json').read_text())
    sample = index[0]
    with np.load(root/sample['file'],allow_pickle=False) as d:
        z = d['embeddings'].copy(); target = d['kv_codes'].copy()
        available = set(d['target_positions'].tolist())
    points = [] if args.preserve_blocks else [p for p in [10,64,256,512,1000,1500,2000] if p in available]
    sections = json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    reports = []
    with NpuBackend(sections,4,root/('canonical_replay_device' if args.preserve_blocks else 'replay_device'),protocol['model_sha256'],
                    serial='58151FDCR00C1E') as b:
        b.reset(protocol['initial_prefix_ids'])
        for image in sample['images']:
            start = image['start']; end = start+image['rows']
            assert b.position == start
            while b.position < end:
                p = b.position
                if p in points:
                    actual = b.query_embeddings(z[p:p+1])[0]
                    delta = actual.astype(float)-target[p]
                    reports.append(dict(position=p,exact=bool(np.all(delta==0)),
                        rmse_lsb=float(np.sqrt(np.mean(delta**2))),
                        max_abs_lsb=float(np.max(np.abs(delta)))))
                next_point = min([x for x in points if p<x<end]+[end])
                count = min(b.batch_capacity,next_point-p)
                part = np.asarray(z[p:p+count],dtype='<f4')
                raw = b.rpc('collect_embeddings',rows=count,position=p,features_hex=part.tobytes().hex())
                if args.preserve_blocks:
                    actual = np.frombuffer(raw,dtype=np.int8).reshape(count,1024)
                    delta = actual.astype(float)-target[p:p+count]
                    reports.append(dict(position=p,rows=count,exact=bool(np.all(delta==0)),
                        rmse_lsb=float(np.sqrt(np.mean(delta**2))),
                        max_abs_lsb=float(np.max(np.abs(delta)))))
            if end < len(z):
                b.commit_embeddings(z[end:end+1])  # real end-of-image constant
            if end+1 < len(z):
                b.commit(255999)
    report = dict(sample=sample['id'],checks=reports,
        all_exact=all(x['exact'] for x in reports),
        scope='one sequence, fresh NPU worker, '+('preserved' if args.preserve_blocks else 'changed')+' causal prefill boundaries; true prefix')
    assert len(reports) > 0
    dump(root/('canonical_replay_control.json' if args.preserve_blocks else 'replay_control.json'),report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
