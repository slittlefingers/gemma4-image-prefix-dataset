"""Produce a compact scientific plot and plain-text reproducibility report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def main():
    s=json.loads((OUT/'summary.json').read_text())
    diag=json.loads((OUT/'diagnostics.json').read_text()) if (OUT/'diagnostics.json').exists() else None
    if diag:s['test_metrics'].update(diag['test_metrics'])
    methods=['separate_chain','smooth_separate_chain','joint','true_e'] if diag else ['separate_chain','direct_ridge','joint','true_e']
    labels=['Separate KV-E-pixels','Smoothed separate chain' if diag else 'Direct KV-pixels ridge','Joint Transformer','True E decoder reference']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    t=s['training'];axes[0].plot([r['epoch'] for r in t],[r['train_mse'] for r in t],label='Training')
    axes[0].plot([r['epoch'] for r in t],[r['validation_mse'] for r in t],label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Pixel MSE',title='Checkpoint chosen by validation');axes[0].legend()
    colors=['#a6a6a6','#77aadd','#1b7837','#ddaa55']
    for ax,metric,title in [(axes[1],'psnr_db','100 held-out HierText images'),(axes[2],'ssim','100 held-out HierText images')]:
        vals=[s['test_metrics'][m][metric] for m in methods]
        ax.bar(np.arange(4),vals,color=colors);ax.set_xticks(np.arange(4),['Separate','Smoothed' if diag else 'Direct ridge','Joint','True E'],rotation=15)
        ax.set(ylabel='PSNR (dB)' if metric=='psnr_db' else 'SSIM',title=title)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}' if metric=='psnr_db' else f'{v:.3f}',ha='center',va='bottom')
        ax.set_ylim(0,max(vals)*1.15)
    fig.suptitle('Fixed 991-token prefix | NPU INT8 layer-5 K/V | 800 train / 100 validation / 100 test')
    fig.savefig(OUT/'results_overview.png',dpi=160);plt.close(fig)
    lines=['Fixed-prefix RASR-style small validation','Dataset Git commit: 955a392',
      'Paper: https://arxiv.org/html/2608.01020v1',
      'Method: coarse visual branch adapted to K/V. One causal Transformer maps all image positions together to latent V (1536); a two-layer MLP maps to latent Z (768); an inverse vision Transformer and U-net-style convolutional decoder produce pixels. Joint image MSE only; no V/Z activation supervision.',
      'This is not an exact reproduction: K/V instead of hidden states, a smaller 4,645,622-parameter network, and 12 epochs at lr=3e-4 instead of the paper settings. No consistency-guided refinement through the original target model.',
      'Data: all 1000 release samples hash-verified; original-image-ID split 800/100/100. Separate whiteboard target excluded from fitting, normalization, and checkpoint choice.',
      'Output: 24 pixels per image token, half the native STB width/height. Targets use 2x2 area reduction. Float16 storage of ordinary train/val/test target tensors; whiteboard evaluation uses float32. Model mixed precision: BF16, unrelated to input NPU INT8 quantization.',
      'Quantization: raw signed INT8 K/V, each 512 dimensions; K scale 0.0011563472216948867, V scale 0.01785714365541935, zero point 0. Per-channel standardization is fit on training observations only.',
      'Comparison: same test images, resolution and metrics for every method. Separate chain is two validation-selected ridge regressions KV->E and E->pixels. Direct ridge maps KV->pixel patches. True-E decoding is a reference decoder result, not an information-theoretic upper bound.',
      'Training duration (neural stage only): %.2f seconds; %d steps.'%(t[-1]['seconds'],t[-1]['step']),
      '', 'Method | Test mean MSE | mean per-image PSNR dB | mean SSIM']
    for m,label in zip(methods,labels):
        r=s['test_metrics'][m];lines.append(f"{label} | {r['mse']:.8f} | {r['psnr_db']:.5f} | {r['ssim']:.5f}")
    lines+=['','Negative control means:']
    for m in ['mean_kv','shuffled_kv']:lines.append(m+': '+json.dumps(s['test_metrics'][m]))
    lines+=['','Paired bootstrap over 100 test images (not training seeds):',json.dumps(s['paired_image_bootstrap_joint_minus_baseline'],indent=2),
            '', 'Independent whiteboard evaluation (single case, not an accuracy estimate):',json.dumps(s['whiteboard_metrics'],indent=2),
            '', 'Checks: independent file hashes; unique source IDs/hashes; fixed-prefix identity; contiguous positions; finite arrays; target source hash absent; target reconstruction saved before truth opened; checkpoint reload max difference '+str(s['reload_max_abs']),
            '', 'Limitations:']+s['limitations']
    lines+=['','Commands:', 'python prepare.py','python run.py --epochs 12','python report.py',
            'python run.py --reuse  # re-evaluate frozen checkpoints, does not train',
            '', 'Artifacts: data_audit.json, download_audit.json (parent), joint_best.pt, ridges.pt, training.json, frozen_before_test.json, summary.json, test_overview.png, whiteboard_comparison.png, whiteboard_kv_only_predictions.pt.',
            'The whiteboard exact text has no independently verified annotation; do not treat guessed text as ground truth.',
            'No comparison to previous COCO scores: data distribution and output resolution changed.']
    if diag:
        lines+=['','Additional diagnostic (added after the first image inspection; checkpoints unchanged):',
                'Gaussian smoothing sigma chosen using validation images only, then applied to test/whiteboard. Chosen sigma=12 output pixels. Final diagnostic metrics use FP32 area-reduced native labels. Cached FP16 label MSE was %.3g.'%diag['cached_fp16_pixel_label_mean_mse'],
                'Joint vs smoothed separate-chain MSE difference: '+json.dumps(diag['bootstrap_joint_minus_smoothed']['smooth_separate_chain']),
                'Whiteboard joint SSIM=%.6f versus training-mean constant RGB SSIM=%.6f. High SSIM on this mostly blank image does NOT demonstrate handwriting recovery.'%(diag['whiteboard_metrics']['joint']['ssim'],diag['whiteboard_metrics']['train_mean_rgb']['ssim']),
                'Visual inspection: coarse blurred brightness/layout recovered on example held-out images, but fine structures/text remain missing. The whiteboard red handwriting is not legible in the KV-only reconstruction.',
                'Joint test gradient-magnitude correlation=%.6f; true-E reference=%.6f. These are Sobel-based diagnostics, not the original paper metric or character accuracy.'%(diag['test_metrics']['joint']['gradient_magnitude_correlation'],diag['test_metrics']['true_e']['gradient_magnitude_correlation']),
                'This pilot supports improvement over the measured linear baselines, including smoothing; it does not isolate the benefit of joint loss from network capacity, nor establish exact image/text recovery.',
                'python diagnostics.py', 'Additional artifacts: diagnostics.json, whiteboard_smoothing_control.png, test_smoothing_control.png.']
    (HERE/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(test=s['test_metrics'],whiteboard=s['whiteboard_metrics'],seconds=t[-1]['seconds']),indent=2))

if __name__=='__main__':main()
