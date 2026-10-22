"""Acquire supervised image embeddings and real phone INT8 KV. No MLX forward.

One sequence file stores shared prefixes; position p uses embeddings[:p],
kv_codes[p], and label embeddings[p]. Prefix length is absolute token count.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kvrecon.common import WORKSPACE, EXPERIMENTS, digest, dump
from kvrecon.npu import NpuBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'data/20261022_pilot_prefill')
    parser.add_argument('--sequences', type=int, default=8)
    parser.add_argument('--source', type=Path, default=WORKSPACE/'kvidence/datasets/shadow_chat/image_chat_100')
    parser.add_argument('--max-prefix', type=int, default=2000)
    parser.add_argument('--serial', default='58151FDCR00C1E')
    args = parser.parse_args()
    if not (args.sequences == 80 or 1 <= args.sequences <= 8) or not 10 <= args.max_prefix <= 2000:
        parser.error('sequences 1..8 or 80, max-prefix 10..2000')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    (out/'sequences').mkdir(exist_ok=True)
    source = args.source
    manifest = json.loads((source/'manifest.json').read_text())
    sections = json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    rng = np.random.default_rng(20261022)
    images = sorted(manifest['samples'], key=lambda x: x['id'])
    assert len({x['media_sha256'] for x in images}) == len(images), 'Duplicate image sources'
    rng.shuffle(images)
    pools = ({split: [x for x in images if x['split'] == split] for split in ('train','validation','test')}
             if all('split' in x for x in images) else dict(train=images[:80], validation=images[80:90], test=images[90:]))
    # Labels are legitimate supervised acquisition data, not victim-evaluation inputs.
    split_by_id = {x['id']: split for split, pool in pools.items() for x in pool}
    prefix = [2, 105, 2364, 109, 255999]
    # Public end-of-image model constant, extracted directly from the same model.
    eoi_path = HERE.parent/'image_chat_0001_layer5/input/end_of_image.npy'
    eoi = np.load(eoi_path).astype(np.float32).reshape(1, 1536)
    protocol = dict(schema=1, status='collecting', source_dataset=manifest['source'],
        source_manifest_sha256=digest(source/'manifest.json'), seed=20261022,
        model_sha256=manifest['capture_identity']['model_sha256'], layer_index=4,
        component_backends=dict(vision_encoder='phone CPU', adapter='phone CPU', decoder='phone NPU'),
        acquisition='real sequential causal prefill_128, retained KV slices; not independent candidate batching',
        image_signature='vision_280', adapter_signature='vision_adapter_280',
        preprocessing='phone STB decode + CatmullRom sRGB resize + Gemma4 patchify, /255',
        prefix_min=10, prefix_max=args.max_prefix, prefix_unit='absolute preceding token positions',
        split_by_image_id=split_by_id, natural_language_prompt=None, system_prompt=None,
        template='BOS + user turn + image start; (image soft tokens + end-of-image + image start)*',
        initial_prefix_ids=prefix, end_of_image_sha256=digest(eoi_path),
        preceding_image_ids=[255999], ple_id_for_continuous_embeddings=0,
        target='post-adapter embedding 1536; pre-adapter 768 also retained',
        prefix_kind='ground truth; recovered-prefix/noise augmentation not yet collected',
        context_kind='constructed multi-image user turn, not natural conversation',
        source_files={str(p.relative_to(WORKSPACE)): digest(p) for p in
            [Path(__file__), WORKSPACE/'kvidence/reconstruction/native/worker.cc',
             WORKSPACE/'kvidence/reconstruction/native/image_collection.h']})
    if (out/'protocol.json').exists():
        old = json.loads((out/'protocol.json').read_text())
        for key in ('model_sha256', 'seed', 'prefix_max', 'source_files', 'source_manifest_sha256'):
            if old[key] != protocol[key]:
                raise ValueError('Resume protocol mismatch: '+key)
    dump(out/'protocol.json', protocol)
    image_records = {}
    index = []
    started = time.perf_counter()
    with NpuBackend(sections, 4, out/'device', protocol['model_sha256'], serial=args.serial) as b:
        dump(out/'quantization.json', b.quant)
        b.reset(prefix)
        prefix_embeddings = np.stack([np.frombuffer(b.rpc('embedding', id=t), dtype='<f4')
                                      for t in prefix])
        assert prefix_embeddings.shape == (5, 1536)
        def collect_block(z):
            rows = []
            for start in range(0, len(z), b.batch_capacity):
                part = np.asarray(z[start:start+b.batch_capacity], dtype='<f4')
                raw = b.rpc('collect_embeddings', rows=len(part),
                            features_hex=part.tobytes().hex(), position=b.position)
                rows.append(np.frombuffer(raw, dtype=np.int8).reshape(len(part),1024).copy())
            return np.concatenate(rows)
        for seq in range(args.sequences):
            train_end, val_end = (60,70) if args.sequences == 80 else (6,7)
            split = 'train' if seq < train_end else ('validation' if seq < val_end else 'test')
            sid = f'{split}_{seq:03d}'
            dest = out/'sequences'/f'{sid}.npz'
            info_path = dest.with_suffix('.json')
            if dest.exists() and info_path.exists():
                info = json.loads(info_path.read_text())
                assert info['sha256'] == digest(dest)
                index.append(info)
                continue
            b.reset([])
            embeddings = []; codes = []; types = []; image_ids = []; local_rows = []; sources = []
            for i, token in enumerate(prefix):
                codes.append(b.commit(token)[0]); embeddings.append(prefix_embeddings[i])
                types.append(0); image_ids.append(-1); local_rows.append(-1)
            pool = pools[split].copy()
            np.random.default_rng(20261022 if args.sequences == 80 else 20261022+seq).shuffle(pool)
            local_seq = seq if split == 'train' else seq-(train_end if split == 'validation' else val_end)
            cursor = local_seq*8 if args.sequences == 80 else 0
            while b.position <= args.max_prefix:
                item = pool[cursor % len(pool)]; cursor += 1
                iid = item['id']; image_file = out/'images'/f'{iid}.npz'
                if not image_file.exists():
                    raw_path = source/item['media']
                    assert digest(raw_path) == item['media_sha256']
                    b.write('collection_image.bin', raw_path.read_bytes())
                    raw = b.rpc('collect_image')
                    reply = b.calls[-1]; n = reply['image_rows']
                    values = np.frombuffer(raw, dtype='<f4').copy()
                    assert values.size == n*(768+1536)
                    u = values[:n*768].reshape(n, 768)
                    z = values[n*768:].reshape(n, 1536)
                    assert np.isfinite(u).all() and np.isfinite(z).all()
                    # Independent native adapter check on a real image row.
                    np.testing.assert_array_equal(b.project_vision(u[:1]), z[:1])
                    np.savez_compressed(image_file, pre_adapter=u, embedding=z)
                    rec = dict(id=iid, split=split, coco_image_id=item['coco_image_id'],
                        source_url=item['source_url'], license=item['license'],
                        source_sha256=item['media_sha256'], feature_sha256=digest(image_file),
                        rows=n, resized_width=reply['resized_width'], resized_height=reply['resized_height'])
                    dump(image_file.with_suffix('.json'), rec)
                    print(f'IMAGE {iid} {split} rows={n}', flush=True)
                rec = json.loads(image_file.with_suffix('.json').read_text())
                assert rec['split'] == split and rec['feature_sha256'] == digest(image_file)
                image_records[iid] = rec
                with np.load(image_file) as data:
                    z = data['embedding'].copy()
                start = b.position
                count = min(len(z), args.max_prefix+1-start)
                # Check query vs committed target at the image boundary.
                query = b.query_embeddings(z[:1])[0]
                if seq == 0 and start == len(prefix):
                    serial_codes = b.commit_embeddings(z[:4])
                    b.reset(prefix)
                    batch_codes = collect_block(z[:4])
                    control = dict(exact=bool(np.array_equal(serial_codes,batch_codes)),
                        rmse_lsb=float(np.sqrt(np.mean((serial_codes.astype(float)-batch_codes)**2))))
                    dump(out/'prefill_control.json', control)
                    np.testing.assert_array_equal(serial_codes,batch_codes)
                    b.reset(prefix)
                kv = collect_block(z[:count])
                np.testing.assert_array_equal(query, kv[0])
                embeddings.extend(z[:count]); codes.extend(kv); types.extend([1]*count)
                image_ids.extend([int(iid)]*count); local_rows.extend(range(count))
                sources.append(dict(image_id=iid, start=start, rows=count, full_rows=len(z)))
                if b.position > args.max_prefix:
                    break
                kv_end = b.commit_embeddings(eoi)[0]
                embeddings.append(eoi[0]); codes.append(kv_end); types.append(0)
                image_ids.append(-1); local_rows.append(-1)
                if b.position <= args.max_prefix:
                    kv_begin = b.commit(255999)[0]
                    embeddings.append(prefix_embeddings[-1]); codes.append(kv_begin); types.append(0)
                    image_ids.append(-1); local_rows.append(-1)
                print(f'PREFIX {sid} length={b.position}', flush=True)
            emb = np.asarray(embeddings, dtype=np.float32)
            kv = np.asarray(codes, dtype=np.int8)
            kind = np.asarray(types, dtype=np.int8)
            positions = np.flatnonzero((kind == 1) & (np.arange(len(kind)) >= 10)
                                       & (np.arange(len(kind)) <= args.max_prefix))
            assert emb.shape == (args.max_prefix+1, 1536)
            assert kv.shape == (len(emb), 1024) and len(positions)
            np.savez_compressed(dest, embeddings=emb, kv_codes=kv, token_types=kind,
                image_ids=np.asarray(image_ids, dtype=np.int32),
                image_rows=np.asarray(local_rows, dtype=np.int32), target_positions=positions)
            info = dict(id=sid, split=split, file=str(dest.relative_to(out)),
                sha256=digest(dest), targets=len(positions), prefix_min=int(positions.min()),
                prefix_max=int(positions.max()), images=sources,
                query_commit_exact=True, seconds=time.perf_counter()-started)
            dump(info_path, info); index.append(info)
            dump(out/'index.json', index)
            print(f'SAVED {sid} targets={len(positions)}', flush=True)
    # Include cached features on resume as well.
    image_records = {p.stem: json.loads(p.read_text()) for p in (out/'images').glob('*.json')}
    dump(out/'image_manifest.json', list(image_records.values()))
    protocol.update(status='complete', sequences=len(index),
                    targets=sum(x['targets'] for x in index), seconds=time.perf_counter()-started)
    dump(out/'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('status','sequences','targets','seconds')}), flush=True)


if __name__ == '__main__':
    main()
