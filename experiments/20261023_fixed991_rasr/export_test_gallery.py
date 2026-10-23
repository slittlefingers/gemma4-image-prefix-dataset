"""Export ten fixed-interval held-out examples using frozen checkpoints.

No training, checkpoint selection, or access to whiteboard truth is performed.
"""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
from run import HERE,ROOT,digest,ridge_apply,patch_image,neural_predict,metrics,save_image,dump
from model import Reconstructor
from diagnostics import blur

def image_tensor(x):
    return Image.fromarray(np.uint8(np.clip(x[0].permute(1,2,0).float().cpu().numpy()*255+.5,0,255)))

def comparison(arrays,scores,sid,path):
    columns=[('original','Reference / 2'),('joint','Joint Transformer'),('smooth_separate_chain','Smoothed separate chain'),
             ('separate_chain','Separate chain'),('true_e','True E reference')]
    w,h=image_tensor(arrays['original']).size;cw=max(w,300);font=ImageFont.load_default(size=18)
    canvas=Image.new('RGB',(5*cw,h+105),'white');draw=ImageDraw.Draw(canvas)
    for j,(key,label) in enumerate(columns):
        x=j*cw;draw.text((x+8,8),label,font=font,fill='black');canvas.paste(image_tensor(arrays[key]),(x+(cw-w)//2,38))
        if key!='original':draw.text((x+8,h+46),f"PSNR {scores[key]['psnr_db']:.2f} | SSIM {scores[key]['ssim']:.3f}",font=font,fill='black')
    draw.text((8,h+78),sid,font=font,fill='black');canvas.save(path)

from sample_report import make_report as make_reports

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,default=HERE/'results');args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    output=HERE/'test_gallery';(output/'comparisons').mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
    frozen=json.loads((HERE/'results/frozen_before_test.json').read_text())
    for key,file in [('joint_checkpoint_sha256','joint_best.pt'),('ridge_checkpoint_sha256','ridges.pt')]:assert digest(args.checkpoints/file)==frozen[key]
    ck=torch.load(args.checkpoints/'joint_best.pt',weights_only=True,map_location='cuda');model=Reconstructor().cuda();model.load_state_dict(ck['state_dict']);model.eval()
    pack=torch.load(args.checkpoints/'ridges.pt',weights_only=True,map_location='cuda');rke,rep=pack['kv_e'],pack['e_pixels']
    audit=json.loads((HERE/'results/data_audit.json').read_text());ids=sorted(audit['split_ids']['test']);assert len(ids)==100
    assert not set(ids).intersection(ck['train_ids']+ck['validation_ids'])
    source={r['id']:r for r in json.loads((ROOT/'index.json').read_text())}
    previous=json.loads((HERE/'results/diagnostics.json').read_text());old={r['id']:r['metrics'] for r in previous['per_image']};assert set(ids)==set(old)
    sigma=previous['smoothing_sigma_output_pixels']['separate_chain'];rows=[];max_metric_delta=0.
    for number,sid in list(enumerate(ids,1))[::10]:
        meta=source[sid];path=ROOT/meta['file'];assert digest(path)==meta['sha256']
        with np.load(path,allow_pickle=False) as f:
            d=dict(k=torch.from_numpy(f['kv_codes'].copy()).cuda(),grid=tuple(int(v) for v in f['grid']))
            e=torch.from_numpy(f['embedding'].copy()).cuda()
            native=torch.from_numpy(f['pixels'].copy()).cuda().permute(2,0,1)[None].float()/255
        y=F.avg_pool2d(native,2);ehat=ridge_apply(d['k'],rke)
        separate=patch_image(ridge_apply(ehat,rep),d['grid'])
        arrays=dict(original=y,joint=neural_predict(model,d,ck['kv_mean'],ck['kv_std']),separate_chain=separate,
                    smooth_separate_chain=blur(separate,sigma),true_e=patch_image(ridge_apply(e,rep),d['grid']))
        scores={k:metrics(p,y) for k,p in arrays.items() if k!='original'}
        for k,values in scores.items():
            for metric,value in values.items():
                delta=abs(value-old[sid][k][metric]);max_metric_delta=max(max_metric_delta,delta)
                assert delta<1e-5,(sid,k,metric,value,old[sid][k][metric])
        stem=f'{number:03d}_{meta["source_image_id"]}'
        comparison(arrays,scores,sid,output/'comparisons'/f'{stem}.png')
        save_image(y,output/'images'/f'{stem}_reference.png');save_image(arrays['joint'],output/'images'/f'{stem}_joint.png')
        files=[f'comparisons/{stem}.png',f'images/{stem}_reference.png',f'images/{stem}_joint.png']
        rows.append(dict(number=number,id=sid,source_image_id=meta['source_image_id'],source_sha256=meta['source_sha256'],
                         sample_sha256=meta['sha256'],stem=stem,width=y.shape[-1],height=y.shape[-2],positions=len(d['k']),
                         metrics=scores,files={file:dict(sha256=digest(output/file),bytes=(output/file).stat().st_size) for file in files}))
        print('EXPORTED',number,flush=True)
    dump(output/'gallery_metrics.json',dict(images=10,test_pool_images=100,selection='IDs sorted ascending, positions 1,11,...,91',checkpoint=frozen,per_image=rows))
    dump(output/'export_audit.json',dict(images=10,comparison_panels=10,reference_pngs=10,joint_pngs=10,test_pool_images=100,
         all_npz_hashes_verified=True,checkpoint_hashes_verified=True,train_validation_ids_disjoint=True,
         selected_test_positions=list(range(1,100,10)),maximum_absolute_metric_difference_from_published_diagnostics=max_metric_delta,
         no_retraining=True,no_checkpoint_selection=True,smoothing_sigma=sigma,metric_reference='results/diagnostics.json',
         checkpoint_sha256=frozen['joint_checkpoint_sha256']))
    make_reports(rows,output);print('COMPLETE',json.dumps(dict(images=10,metric_max_delta=max_metric_delta)),flush=True)

if __name__=='__main__':main()
