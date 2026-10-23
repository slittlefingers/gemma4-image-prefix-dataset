"""Read fixed-prefix INT8 KV using the shipped quantization.json metadata."""
import argparse
import json
from pathlib import Path
import numpy as np


def dequantize_kv(codes, quantization):
    values = np.asarray(codes, dtype=np.float32)
    if values.shape[-1] != sum(part['width'] for part in quantization):
        raise ValueError('KV final dimension does not match quantization metadata')
    result = np.empty_like(values)
    start = 0
    for part in quantization:  # First K, then V; preserve the metadata order.
        stop = start + part['width']
        result[..., start:stop] = (values[..., start:stop] - np.float32(part['zero_point'])) * np.float32(part['scale'])
        start = stop
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=Path, help='Training samples/*.npz or important test observed.npz')
    parser.add_argument('--quantization', type=Path, required=True)
    args = parser.parse_args()
    quantization = json.loads(args.quantization.read_text())
    with np.load(args.sample, allow_pickle=False) as data:
        codes = data['kv_codes'] if 'kv_codes' in data else data['codes']
        values = dequantize_kv(codes, quantization)
        print('raw:', codes.shape, codes.dtype)
        print('dequantized:', values.shape, values.dtype)
        print('K:', values[..., :quantization[0]['width']].shape)
        print('V:', values[..., quantization[0]['width']:].shape)
