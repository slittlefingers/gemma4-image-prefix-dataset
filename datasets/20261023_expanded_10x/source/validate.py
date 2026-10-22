"""Independent structural and provenance audit of the completed acquisition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data', type=Path)
    root = p.parse_args().data
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['status'] in ('complete','complete_fixed_prefill_protocol')
    index = json.loads((root/'index.json').read_text())
    quant = json.loads((root/'quantization.json').read_text())
    assert sum(x['width'] for x in quant) == 1024
    assert all(x['dtype'] == 'int8' and x['scale'] > 0 for x in quant)
    split_images = {}; reports = []; bins = [(10,31),(32,63),(64,127),(128,255),
        (256,511),(512,999),(1000,1499),(1500,2000)]
    for row in index:
        path = root/row['file']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as d:
            z, kv, types, ids, local, pos = [d[k] for k in
                ('embeddings','kv_codes','token_types','image_ids','image_rows','target_positions')]
            assert z.shape == (protocol['prefix_max']+1,1536) and z.dtype == np.float32
            assert kv.shape == (len(z),1024) and kv.dtype == np.int8
            assert np.isfinite(z).all()
            wanted = np.flatnonzero((types == 1) & (np.arange(len(z)) >= 10))
            np.testing.assert_array_equal(pos, wanted)
            assert len(pos) == row['targets'] and row['query_commit_exact']
            for item in row['images']:
                iid = item['image_id']; split_images.setdefault(row['split'],set()).add(iid)
                assert protocol['split_by_image_id'][iid] == row['split']
                feature_file = root/'images'/f'{iid}.npz'
                meta = json.loads(feature_file.with_suffix('.json').read_text())
                assert sha(feature_file) == meta['feature_sha256']
                a = item['start']; b = a+item['rows']
                with np.load(feature_file, allow_pickle=False) as image:
                    np.testing.assert_array_equal(z[a:b],image['embedding'][:b-a])
                    assert image['pre_adapter'].shape == (item['full_rows'],768)
                assert np.all(ids[a:b] == int(iid)) and np.all(types[a:b] == 1)
                np.testing.assert_array_equal(local[a:b],np.arange(b-a))
            reports.append(dict(id=row['id'],split=row['split'],targets=len(pos),
                length_bins={f'{a}-{b}':int(np.sum((pos>=a)&(pos<=b))) for a,b in bins}))
    splits = list(split_images)
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            assert not split_images[a] & split_images[b]
    report = dict(valid=True,sequences=len(index),targets=sum(r['targets'] for r in reports),
        unique_images={s:len(v) for s,v in split_images.items()}, rows=reports,
        checks=['file hashes','finite true labels','image-exclusive splits','target positions',
                'per-image label equality','raw INT8 dtype and quantization',
                'recorded real query/commit exact checks'],
        limitations=['ground-truth prefixes only','constructed multi-image turns',
                     'no claim of end-to-end chat runtime equivalence'])
    canonical = root/'canonical_replay_control.json'
    if canonical.exists():
        replay = json.loads(canonical.read_text())
        assert replay['all_exact'] and all(x['rmse_lsb']==0 for x in replay['checks'])
        report['canonical_replay'] = dict(all_exact=True,
            image_positions=sum(x['rows'] for x in replay['checks']),sample=replay['sample'])
        report['limitations'].append('changed prefill/query paths are not numerically identical; see replay_control.json')
    assert report['targets'] == protocol['targets']
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
