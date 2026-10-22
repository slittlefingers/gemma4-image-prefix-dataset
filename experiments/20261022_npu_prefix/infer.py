"""Predict one embedding from prefix[N,1536] and raw int8 kv[1024] in an NPZ."""
import argparse
from pathlib import Path

import numpy as np
import torch

from train import Inverse


def load_model(path, device):
    checkpoint=torch.load(path,weights_only=True,map_location=device)
    state=checkpoint['state_dict'];metadata=checkpoint['metadata']
    ridge={name:state[name] for name in ['xm','xs','ym','ys','weight','bias']}
    ridge['kind']=metadata['ridge_kind']
    model=Inverse(ridge,metadata['mode']).to(device)
    model.load_state_dict(state);model.eval()
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True,help='NPZ with prefix and kv arrays only')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(args.input,allow_pickle=False) as f:
        prefix=f['prefix'].copy();kv=f['kv'].copy()
    assert prefix.ndim==2 and prefix.shape[1]==1536 and 1<=len(prefix)<=2000
    assert kv.shape==(1024,) and kv.dtype==np.int8
    assert np.isfinite(prefix).all()
    model=load_model(args.checkpoint,device)
    with torch.no_grad():
        prediction=model(torch.from_numpy(kv)[None,None].to(device),
                         torch.tensor([len(prefix)],device=device),
                         torch.from_numpy(prefix).float()[None].to(device))[0,0]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,prediction.cpu().numpy())
    print(f'Saved {args.output}: shape=(1536,), position={len(prefix)}, dtype=float32')


if __name__=='__main__':main()
