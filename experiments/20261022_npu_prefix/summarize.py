"""Report all three predetermined seeds, with no test-driven model selection."""
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def main():
    reports=[json.loads((HERE/f'results_seed{s}/summary.json').read_text()) for s in [20261022,20261023,20261024]]
    r=reports[0];rows=[]
    for name in r['results']:
        values=[rr['results'][name]['overall'] for rr in reports]
        rows.append(dict(name=name,mean={k:float(np.mean([v[k] for v in values])) for k in values[0]},
                         rmse_seed_std=float(np.std([v['rmse'] for v in values],ddof=1))))
    result=dict(seeds=[x['seed'] for x in reports],test_images=8,test_sequences=1,test_positions=1977,
                aggregate=rows,training=[dict(seed=x['seed'],models={k:{f:v[f] for f in ['best_step','last_step_train_subset_relative_rmse','last_step_train_subset_mse','last_step_prefix_projection_weight_change','seconds','parameters']} for k,v in x['training'].items()}) for x in reports],
                environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
                limits=r['limits'])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=['Gemma4 real-NPU image-prefix inverse pilot (2026-10-22)',
          'Source: https://github.com/slittlefingers/gemma4-image-prefix-dataset',
          'Commit: 8e4005d4dd9fbc314628f0b3c7ca62d2cb59ac91',
          'Release: v2026.10.22-pilot',
          'Archive SHA256: e65d891a90eb626676a35705219980e94d4f6c6cd736dd9260892146f317aa82',
          '40 training images / 8 validation / 8 test; 6 / 1 / 1 sequences.',
          '11862 / 1977 / 1977 target positions; original image-disjoint split retained.',
          'All archive, sequence, feature hashes and embedding labels verified.',
          'Input: raw current INT8 K/V (512+512), absolute position, strict prefix embeddings.',
          'Output: current 1536-dimensional post-adapter embedding.',
          'No current/future embeddings or other-position KVs accessible to the prediction.',
          'No target model weights/gradients/phone used to train the inverse.',
          'K/V ridge alpha selected on validation, followed by 2000 AdamW steps per neural model.',
          'Two 192-wide blocks, 4 attention heads, 64 queries/step, equal prefix-length-bin sampling.',
          'Three seeds; model selection exclusively validation embedding MSE.',
          '', 'TEST: mean across seeds; RMSE is per embedding dimension, not KV LSB.',
          'Method                      RMSE       Relative RMSE     Cosine']
    for row in rows:
        m=row['mean'];text.append(f"{row['name']:28s} {m['rmse']:.6f}   {100*m['relative_rmse']:.3f}%          {m['mean_cosine']:.6f}")
    text.extend(['','TRAINING DIAGNOSTICS'])
    for x in result['training']:
        text.append(json.dumps(x))
    text.extend(['','Interpretation:'])
    all_zero=all(x['training']['prefix']['best_step']==0 for x in reports)
    if all_zero:
        text.extend(['All prefix models selected step 0: the ridge initializer. Attention training did not improve held-out validation.',
                     'Identical teacher/rollout results do NOT demonstrate robustness to prediction errors: selected models ignore the prefix.',
                     'Final trained checkpoints retained as prefix_last.pt; best-validation checkpoint is prefix_model.pt.'])
    text.extend(['','Test prefix-length bins (ridge KV):'])
    for b,m in r['results']['ridge_kv']['prefix_bins'].items():
        text.append(f"{b}: RMSE={m['rmse']:.6f}, relative RMSE={100*m['relative_rmse']:.3f}%")
    text.extend(['','Limitations:']+r['limits'])
    text.extend(['','Run from the repository root:',
                 'python experiments/20261022_npu_prefix/train.py --steps 2000 --seed 20261022 --out experiments/20261022_npu_prefix/results_seed20261022',
                 'python experiments/20261022_npu_prefix/infer.py --checkpoint <prefix_model.pt> --input <input.npz> --output <embedding.npy>',
                 'Input NPZ: prefix float32 array [p,1536], kv raw int8 array [1024].',
                 'Original 60-token synthetic-image decoder is not applied to these variable 256-280-token natural images.'])
    (HERE/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    names=['mean','ridge_k','ridge_v','ridge_kv','prefix_true']
    selected=[next(x for x in rows if x['name']==n) for n in names]
    axes[0].bar(['Mean','K ridge','V ridge','KV ridge','Prefix selected'],[x['mean']['relative_rmse']*100 for x in selected],color=['gray','#94b8d8','#729cc4','#305e89','#ce9456'])
    axes[0].set_ylabel('Embedding relative RMSE (%)');axes[0].tick_params(axis='x',rotation=25)
    axes[0].set_title('Held-out test: 8 images / 1 sequence')
    bins=r['results']['ridge_kv']['prefix_bins']
    axes[1].plot(range(len(bins)),[m['relative_rmse']*100 for m in bins.values()],marker='o')
    axes[1].set_xticks(range(len(bins)),list(bins),rotation=50);axes[1].set_ylim(0,100)
    axes[1].set_title('KV ridge by absolute prefix length');axes[1].set_ylabel('Embedding relative RMSE (%)')
    for rr in reports:
        t=rr['training']['prefix']['trace']
        axes[2].plot([x['step'] for x in t],[x['validation']['relative_rmse']*100 for x in t],label=str(rr['seed']))
    axes[2].set_title('Prefix attention: validation');axes[2].set_ylabel('Embedding relative RMSE (%)');axes[2].set_xlabel('Training step');axes[2].legend(fontsize=8)
    fig.suptitle('Real NPU observations; no NPU re-forward error measured',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'training_results.png',dpi=160);plt.close(fig)
    print('\n'.join(text))


if __name__=='__main__':main()
