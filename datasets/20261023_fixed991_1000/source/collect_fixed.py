"""Real phone CPU vision / NPU decoder acquisition under one frozen text prefix."""
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import WORKSPACE,EXPERIMENTS,digest,dump
from kvrecon.npu import NpuBackend

MODEL_SHA='af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=WORKSPACE/'kvidence/datasets/hiertext/manifest.json');ap.add_argument('--out',type=Path,default=HERE/'data/20261023_fixed991');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--include-target',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'samples').mkdir(exist_ok=True)
    pm=json.loads((HERE/'protocol/prefix.json').read_text());prefix=pm['prefix_ids']
    manifest=json.loads(args.manifest.read_text());source_root=args.manifest.parent
    records=manifest['samples'];records=sorted(records,key=lambda r:(not r.get('pilot',False),{'train':0,'validation':1,'test':2}[r['split']],r['image_id']))
    if args.limit:records=records[:args.limit]
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    suffix=Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json')).encode(pm['suffix_question']+'<turn|>\n<|turn>model\n',add_special_tokens=False).ids
    eoi=np.load(HERE.parent/'image_chat_0001_layer5/input/end_of_image.npy').astype('<f4').reshape(1,1536)
    source_files=[Path(__file__),HERE/'freeze_prefix.py',WORKSPACE/'kvidence/reconstruction/native/worker.cc',WORKSPACE/'kvidence/reconstruction/native/image_collection.h']
    protocol=dict(schema=1,status='collecting',prefix_tokens=len(prefix),prefix_sha256=pm['prefix_sha256'],prefix_metadata=pm,model_sha256=MODEL_SHA,layer_index=4,source_manifest=str(args.manifest),source_manifest_sha256=digest(args.manifest),source_files={str(p.relative_to(WORKSPACE)):digest(p) for p in source_files},acquisition='fixed-prefix snapshot, causal prefill_128 image blocks, native STB pixel targets',component_backends=dict(vision_encoder='phone CPU',adapter='phone CPU',decoder='phone NPU'),target_in_training=False,planned_samples=len(records),suffix_ids=suffix,suffix_execution='question recorded for every sample; only private target executes EOI/question, since later suffix cannot affect earlier visual KV under causal masking',target_capture_requested=args.include_target)
    pp=args.out/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        for k in ['prefix_sha256','model_sha256','layer_index','source_manifest_sha256','source_files']:assert old[k]==protocol[k],k
    dump(pp,protocol)
    (args.out/'source').mkdir(exist_ok=True)
    for p in source_files:shutil.copy2(p,args.out/'source'/p.name)
    shutil.copy2(HERE/'protocol/prefix.txt',args.out/'prefix.txt');shutil.copy2(HERE/'protocol/prefix.json',args.out/'prefix.json')
    started=time.perf_counter();index=[]
    with NpuBackend(sections,4,args.out/'device'/f'run_{time.time_ns()}',MODEL_SHA,serial='58151FDCR00C1E') as b:
        dump(args.out/'quantization.json',b.quant)
        if b.connection is not None:b.connection.settimeout(900)
        tick=time.perf_counter();b.reset(prefix);print('PREFIX_READY',len(prefix),time.perf_counter()-tick,flush=True)
        b.rpc('save_acquisition_prefix',training_acquisition=True)
        def restore():
            b.rpc('restore_acquisition_prefix',training_acquisition=True);assert b.position==len(prefix)
        def decode(z):
            rows=[]
            for i in range(0,len(z),128):
                part=np.asarray(z[i:i+128],dtype='<f4')
                raw=b.rpc('collect_embeddings',rows=len(part),position=b.position,features_hex=part.tobytes().hex())
                rows.append(np.frombuffer(raw,dtype=np.int8).reshape(len(part),1024).copy())
            return np.concatenate(rows)
        def image(path):
            b.write('collection_image.bin',path.read_bytes());raw=b.rpc('collect_image',include_pixels=True);r=b.calls[-1];n=r['image_rows'];size=n*(768+1536)*4
            v=np.frombuffer(raw[:size],dtype='<f4').copy();u=v[:n*768].reshape(n,768);z=v[n*768:].reshape(n,1536)
            h,w=r['resized_height'],r['resized_width'];assert len(raw)==size+h*w*3
            rgb=np.frombuffer(raw[size:],dtype=np.uint8).reshape(h,w,3).copy();assert n==(h//48)*(w//48)
            assert np.isfinite(u).all() and np.isfinite(z).all()
            return u,z,rgb
        def acquire(path,complete_suffix=False):
            restore();u,z,pixels=image(path);kv=decode(z)
            assert b.position==len(prefix)+len(z)
            # Complete the same user request; suffix is not an input to visual recovery.
            tail=np.empty((0,1024),dtype=np.int8)
            if complete_suffix:
                b.commit_embeddings(eoi)
                tail=np.concatenate([b.commit(t) for t in suffix])
            return dict(pre_adapter=u,embedding=z,pixels=pixels,kv_codes=kv,positions=np.arange(len(prefix),len(prefix)+len(z),dtype=np.int32),grid=np.array([pixels.shape[0]//48,pixels.shape[1]//48],dtype=np.int32),suffix_kv_codes=tail)
        # Validate a new native state path before processing the dataset.
        first=records[0];first_path=Path(first.get('absolute_file',source_root/first['file']));assert digest(first_path)==first['sha256']
        control=acquire(first_path);restore();again=decode(control['embedding']);np.testing.assert_array_equal(control['kv_codes'],again)
        # Recompute the full public prefix independently, rather than trusting snapshot restore.
        b.reset(prefix);fresh=decode(control['embedding']);np.testing.assert_array_equal(control['kv_codes'],fresh)
        restore()
        audit=dict(valid=True,sample=first['image_id'],rows=len(fresh),snapshot_replay_exact=True,fresh_prefix_replay_exact=True,rmse_lsb=0,pixel_source='native STB resized image, recovered losslessly from /255 input patches')
        dump(args.out/'acquisition_control.json',audit);print('CONTROL_PASS',audit,flush=True)
        if args.include_target:
            plan=json.loads((HERE/'experiment_plan.json').read_text());target=Path(plan['target']['candidate_file']);assert digest(target)==plan['target']['sha256']
            folder=HERE/'private_target/capture_fixed991';folder.mkdir(exist_ok=True)
            data=acquire(target,complete_suffix=True);np.savez_compressed(folder/'observed.npz',codes=data['kv_codes'],positions=data['positions'],grid=data['grid'])
            np.savez_compressed(folder/'evaluation_only_truth.npz',**{k:v for k,v in data.items() if k!='kv_codes'})
            dump(folder/'metadata.json',dict(prefix_sha256=pm['prefix_sha256'],model_sha256=MODEL_SHA,layer_index=4,source_sha256=digest(target),observation_sha256=digest(folder/'observed.npz'),grid=data['grid'].tolist(),image_rows=len(data['embedding']),public_prefix_ids=prefix,training_excluded=True,publication_authorized=False))
            print('PRIVATE_TARGET_CAPTURED',len(data['embedding']),flush=True)
        for n,r in enumerate(records,1):
            sid=f"{r.get('source_dataset','hiertext')}_{r['split']}_{r['image_id']}";dest=args.out/'samples'/f'{sid}.npz';meta=dest.with_suffix('.json')
            if dest.exists() and meta.exists():
                row=json.loads(meta.read_text());assert row['sha256']==digest(dest);index.append(row);continue
            src=Path(r.get('absolute_file',source_root/r['file']));assert digest(src)==r['sha256']
            data=acquire(src);tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(dest)
            row=dict(id=sid,source_dataset=r.get('source_dataset','HierText'),source_image_id=r['image_id'],source_sha256=r['sha256'],split=r['split'],pilot=r.get('pilot',False),file=str(dest.relative_to(args.out)),sha256=digest(dest),grid=data['grid'].tolist(),image_rows=len(data['embedding']),image_start=len(prefix),prefix_sha256=pm['prefix_sha256'],seconds=time.perf_counter()-started)
            dump(meta,row);index.append(row);dump(args.out/'index.json',index)
            dump(args.out/'progress.json',dict(status='collecting',completed=len(index),planned=len(records),seconds=time.perf_counter()-started,last_sample=sid))
            print('SAVED',n,'/',len(records),sid,'grid',row['grid'],'seconds',round(row['seconds'],2),flush=True)
    protocol.update(status='complete',samples=len(index),targets=sum(r['image_rows'] for r in index),seconds=time.perf_counter()-started)
    dump(pp,protocol);dump(args.out/'index.json',index);dump(args.out/'progress.json',dict(status='complete',completed=len(index),planned=len(records),seconds=protocol['seconds']))
    print('COMPLETE',len(index),flush=True)

if __name__=='__main__':main()
