"""Audit complete fixed-prefix acquisition; target is outside this dataset."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);root=ap.parse_args().data
 protocol=json.loads((root/'protocol.json').read_text());assert protocol['status']=='complete'
 index=json.loads((root/'index.json').read_text());assert len(index)==protocol['planned_samples']==protocol['samples']
 control=json.loads((root/'acquisition_control.json').read_text());assert control['valid'] and control['snapshot_replay_exact'] and control['fresh_prefix_replay_exact']
 target_sha=json.loads((Path(__file__).resolve().parent/'experiment_plan.json').read_text())['target']['sha256']
 p=protocol['prefix_tokens'];by_split={};seen=set();hashes=set();rows=0
 for r in index:
  assert r['id'] not in seen and r['source_sha256'] not in hashes;seen.add(r['id']);hashes.add(r['source_sha256'])
  path=root/r['file'];assert sha(path)==r['sha256'];assert r['prefix_sha256']==protocol['prefix_sha256'] and r['image_start']==p
  with np.load(path,allow_pickle=False) as d:
   n=r['image_rows'];grid=d['grid'];pixels=d['pixels'];assert d['kv_codes'].shape==(n,1024) and d['kv_codes'].dtype==np.int8
   assert d['embedding'].shape==(n,1536) and d['embedding'].dtype==np.float32
   assert d['pre_adapter'].shape==(n,768) and d['pre_adapter'].dtype==np.float32
   assert np.isfinite(d['embedding']).all() and np.isfinite(d['pre_adapter']).all()
   assert pixels.shape==(int(grid[0])*48,int(grid[1])*48,3) and pixels.dtype==np.uint8 and int(np.prod(grid))==n
   np.testing.assert_array_equal(d['positions'],np.arange(p,p+n))
   assert grid.tolist()==r['grid']
  x=by_split.setdefault(r['split'],dict(images=0,positions=0,sources={}));x['images']+=1;x['positions']+=n;x['sources'][r['source_dataset']]=x['sources'].get(r['source_dataset'],0)+1;rows+=n
 assert rows==protocol['targets']
 assert target_sha not in hashes,'Private target leaked into dataset'
 summary=dict(valid=True,images=len(index),image_positions=rows,prefix_tokens=p,split_summary=by_split,source_hash_overlap=0,target_sha256_overlap=0,validator_sha256=sha(Path(__file__)),checks=['array file SHA256','unique source image IDs and byte hashes','fixed prefix identity','native image grid and pixel dimensions','finite FP32 labels','raw INT8 KV','contiguous image positions','snapshot and fresh-prefix exact replay'],limits=['specific constructed fixed conversation','candidate-query or alternate prefill paths not certified','target image is a separate private evaluation case','validation/test source domain is HierText'])
 (root/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
