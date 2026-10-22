"""Add explicit last-step diagnostic results without retraining or reselecting."""
import json
import numpy as np
import torch
from train import HERE, ROOT, load_data, predict, grouped_metrics, save_json
from infer import load_model


def main():
    torch.set_num_threads(4)
    data,_=load_data(ROOT,'cuda');d=next(x for x in data if x['split']=='test')
    p=d['target_positions'];target=d['embeddings'][p]
    for seed in [20261022,20261023,20261024]:
        out=HERE/f'results_seed{seed}';model=load_model(out/'prefix_last.pt','cuda')
        with torch.no_grad():
            predictions=dict(prefix_last_true=predict(model,d),prefix_last_rollout=model.rollout(d),
                             prefix_last_mean=predict(model,d,model.ym[None].expand_as(d['embeddings'])))
        report=json.loads((out/'summary.json').read_text())
        stored=torch.load(out/'test_predictions.pt',weights_only=True)
        for name,pr in predictions.items():
            report['results'][name]=grouped_metrics(pr,target,d)
            stored['predictions'][name]=pr.cpu()
        save_json(out/'summary.json',report);torch.save(stored,out/'test_predictions.pt')
        print(seed,{k:report['results'][k]['overall'] for k in predictions},flush=True)
    # Standalone inference example has no current/future embedding or target.
    q=1000
    np.savez(HERE/'inference_input.npz',prefix=d['embeddings'][:q].cpu().numpy(),kv=d['kv_codes'][q].cpu().numpy())


if __name__=='__main__':main()
