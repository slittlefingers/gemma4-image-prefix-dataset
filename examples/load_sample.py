# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Read one supervised pair without exposing current/future labels as input."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_sample(root, sequence, position):
    path = root/'sequences'/f'{sequence}.npz'
    with np.load(path, allow_pickle=False) as data:
        if position not in data['target_positions']:
            raise ValueError('Position is not a collected image target')
        prefix = data['embeddings'][:position].copy()
        codes = data['kv_codes'][position].copy()
        label = data['embeddings'][position].copy()
        image_id = int(data['image_ids'][position])
        image_row = int(data['image_rows'][position])
    quant = json.loads((root/'quantization.json').read_text())
    scale = np.concatenate([np.full(q['width'],q['scale']) for q in quant])
    zero = np.concatenate([np.full(q['width'],q['zero_point']) for q in quant])
    original_units = (codes.astype(np.float64)-zero)*scale
    with np.load(root/'images'/f'{image_id:04d}.npz',allow_pickle=False) as image:
        pre_adapter = image['pre_adapter'][image_row].copy()
    return dict(prefix_embeddings=prefix,kv_codes=codes,kv_dequantized=original_units,
                label_embedding=label,label_pre_adapter=pre_adapter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/20261022_pilot_prefill'))
    parser.add_argument('--sequence',default='train_000')
    parser.add_argument('--position',type=int,default=1000)
    args = parser.parse_args()
    sample = load_sample(args.data,args.sequence,args.position)
    for name, value in sample.items():
        print(f'{name}: shape={value.shape}, dtype={value.dtype}')
