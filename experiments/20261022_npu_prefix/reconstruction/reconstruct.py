"""Real phone image embeddings -> pixels, using disjoint COCO image splits.

Our supervised spatial decoder adaptation, not official paper code.
No diffusion/generative prior, original target pixels, or target-model gradients
are supplied when reconstructing held-out observations.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train import ROOT, load_data, predict_ridge
from infer import load_model


def emit(name,value):print(name,json.dumps(value),flush=True)


class Residual(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(n,n,3,padding=1),nn.GELU(),nn.Conv2d(n,n,3,padding=1))
    def forward(self,x):return x+.2*self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Conv2d(1536,96,1),nn.GELU(),Residual(96),Residual(96))
        layers=[];n=96
        for factor,width in [(3,64),(2,32),(2,16)]:
            layers.extend([nn.Conv2d(n,width*factor**2,3,padding=1),nn.PixelShuffle(factor),nn.GELU(),Residual(width)])
            n=width
        self.up=nn.Sequential(*layers);self.output=nn.Conv2d(16,3,3,padding=1)
        self.coarse=nn.Conv2d(1536,3,1)
        nn.init.zeros_(self.coarse.weight);nn.init.zeros_(self.coarse.bias)
        nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)
    def forward(self,e):
        coarse=F.interpolate(self.coarse(e),scale_factor=12,mode='bilinear',align_corners=False)
        return torch.sigmoid(coarse+self.output(self.up(self.input(e))))


def metrics(pred,target,mask=None):
    a=pred.astype(np.float64);b=target.astype(np.float64)
    if mask is None:mask=np.ones(a.shape[:2],dtype=bool)
    mse=float(((a-b)**2)[mask].mean())
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma
    vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    ss=((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    inner=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    inner[:5]=False;inner[-5:]=False;inner[:,:5]=False;inner[:,-5:]=False
    return dict(mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),ssim=float(ss[inner].mean()))


def save_image(a,path):Image.fromarray(np.round(np.clip(a,0,1)*255).astype(np.uint8)).save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);args=ap.parse_args()
    out=HERE/'results';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261022);np.random.seed(20261022)
    manifest=json.loads((ROOT/'image_manifest.json').read_text());items={}
    for m in manifest:
        with np.load(ROOT/'images'/f"{m['id']}.npz",allow_pickle=False) as d:e=torch.from_numpy(d['embedding'].copy()).cuda()
        gh=m['resized_height']//48;gw=m['resized_width']//48;assert gh*gw==len(e)
        im=Image.open(HERE/'images'/f"{m['id']}.jpg").convert('RGB')
        # Quarter native input size. PIL Lanczos reference is not byte-identical to STB.
        pixels=np.asarray(im.resize((gw*12,gh*12),Image.Resampling.LANCZOS)).copy()/255.
        y=torch.from_numpy(pixels).float().cuda().permute(2,0,1)[None]
        items[m['id']]=dict(**m,e=e,y=y,grid=(gh,gw),pixels=pixels)
    train=[d for d in items.values() if d['split']=='train'];val=[d for d in items.values() if d['split']=='validation']
    all_e=torch.cat([d['e'] for d in train]);mean=all_e.mean(0);std=all_e.std(0).clamp_min(.05)
    for d in items.values():d['x']=((d['e']-mean)/std).T.reshape(1,1536,*d['grid'])
    # Linear patch decoder is an independently fitted, validation-selected baseline.
    xx=torch.cat([d['x'].flatten(2).transpose(1,2)[0] for d in train])
    yy=torch.cat([F.unfold(d['y'],12,stride=12)[0].T for d in train])
    xm=xx.mean(0);ym=yy.mean(0);xd=(xx-xm).double();yd=(yy-ym).double()
    gram=xd.T@xd/len(xd);rhs=xd.T@yd/len(xd);ev,vec=torch.linalg.eigh(gram)
    proj=vec.T@rhs;ridge_best=math.inf;ridge_trace=[]
    def ridge_image(d,w,b):
        rows=d['x'].flatten(2).transpose(1,2)[0]@w+b
        return F.fold(rows.T[None],output_size=d['y'].shape[-2:],kernel_size=12,stride=12).clamp(0,1)
    for alpha in [.001,.01,.1,1.,10.]:
        w=(vec@(proj/(ev[:,None]+alpha))).float();b=ym-xm@w
        vm=float(np.mean([float(F.mse_loss(ridge_image(d,w,b),d['y'])) for d in val]))
        ridge_trace.append(dict(alpha=alpha,val_mse=vm))
        if vm<ridge_best:ridge_best=vm;ridge_w=w.clone();ridge_b=b.clone();ridge_alpha=alpha
    emit('RIDGE_DECODER',dict(best_alpha=ridge_alpha,validation_mse=ridge_best))
    del gram,rhs,ev,vec,proj,xx,yy,xd,yd
    model=Decoder().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
    best=math.inf;trace=[];start=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train();xs=[];ys=[]
        # Random spatial crops from training images only, with center loss to reduce edge mismatch.
        for _ in range(8):
            d=train[int(torch.randint(len(train),()).item())];gh,gw=d['grid'];size=min(10,gh,gw)
            y=int(torch.randint(gh-size+1,()).item());x=int(torch.randint(gw-size+1,()).item())
            xs.append(d['x'][:,:,y:y+size,x:x+size]);ys.append(d['y'][:,:,y*12:(y+size)*12,x*12:(x+size)*12])
        x=torch.cat(xs);y=torch.cat(ys)
        opt.zero_grad(set_to_none=True);pred=model(x);loss=F.mse_loss(pred[:,:,12:-12,12:-12],y[:,:,12:-12,12:-12])
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        for g in opt.param_groups:g['lr']=3e-4*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        opt.step()
        if step==1 or step%100==0 or step==args.steps:
            model.eval()
            with torch.no_grad():vm=float(np.mean([float(F.mse_loss(model(d['x']),d['y'])) for d in val]))
            if vm<best:best=vm;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(step=step,train_mse=float(loss.detach()),val_mse=vm,best_step=best_step,seconds=time.perf_counter()-start)
            trace.append(row);emit('DECODER_TRAIN',row)
    model.load_state_dict(best_state);model.eval().requires_grad_(False)
    selected='cnn' if best<ridge_best else 'ridge'
    ckpt=dict(state_dict=best_state,mean=mean.cpu(),std=std.cpu(),ridge_weight=ridge_w.cpu(),ridge_bias=ridge_b.cpu(),
              selected=selected,cnn_best_step=best_step,cnn_validation_mse=best,ridge_validation_mse=ridge_best,
              ridge_alpha=ridge_alpha,ridge_trace=ridge_trace,trace=trace,
              train_ids=[d['id'] for d in train],validation_ids=[d['id'] for d in val],seed=20261022)
    torch.save(ckpt,out/'image_decoder.pt')
    @torch.no_grad()
    def decode(e,gh,gw,kind=selected):
        x=((e-mean)/std).T.reshape(1,1536,gh,gw)
        if kind=='cnn':r=model(x)
        else:
            rows=x.flatten(2).transpose(1,2)[0]@ridge_w+ridge_b
            r=F.fold(rows.T[None],(gh*12,gw*12),12,stride=12).clamp(0,1)
        return r[0].permute(1,2,0).cpu().numpy()
    # Frozen decoder: test observations and inverse predictions first used here.
    sequences,audit=load_data(ROOT,'cuda');test=next(d for d in sequences if d['split']=='test')
    index=next(d for d in json.loads((ROOT/'index.json').read_text()) if d['split']=='test')
    ridge=torch.load(HERE.parent/'results_seed20261022/ridges.pt',weights_only=True,map_location='cuda')['kv']
    inverse=load_model(HERE.parent/'results_seed20261022/prefix_last.pt','cuda')
    with torch.no_grad():
        predicted=predict_ridge(test['kv_codes'],ridge)
        rollout=inverse.rollout(test,return_full=True)
    torch.save(dict(best_kv_embedding=predicted.cpu(),prefix_rollout_embedding=rollout.cpu()),out/'full_sequence_predictions.pt')
    rows=[];panels=[];generator=torch.Generator(device='cuda').manual_seed(71)
    for entry in index['images']:
        iid=entry['image_id'];d=items[iid];gh,gw=d['grid'];count=entry['rows'];s=entry['start'];n=gh*gw
        er=mean[None].expand(n,-1).clone();ep=er.clone()
        er[:count]=predicted[s:s+count];ep[:count]=rollout[s:s+count]
        arrays=dict(original=d['pixels'],true_embedding=decode(d['e'],gh,gw),
                    best_kv_inverse=decode(er,gh,gw),prefix_rollout=decode(ep,gh,gw),
                    mean_embedding=decode(mean[None].expand(n,-1),gh,gw),
                    shuffled_true_embedding=decode(d['e'][torch.randperm(n,generator=generator,device='cuda')],gh,gw))
        valid=np.arange(n).reshape(gh,gw)<count;mask=np.repeat(np.repeat(valid,12,0),12,1)
        result=dict(image_id=iid,coco_image_id=d['coco_image_id'],observed_rows=count,full_rows=n,
                    complete=count==n,grid=[gh,gw],output_size=[gw*12,gh*12],
                    metrics={name:metrics(a,d['pixels'],mask) for name,a in arrays.items() if name!='original'})
        rows.append(result)
        for name,a in arrays.items():
            displayed=a.copy()
            if name in ['best_kv_inverse','prefix_rollout']:displayed[~mask]=.65
            save_image(displayed,out/f'{iid}_{name}.png')
            arrays[name]=displayed
        panels.append((iid,arrays,count==n))
    complete=[r for r in rows if r['complete']]
    aggregate={name:{k:float(np.mean([r['metrics'][name][k] for r in complete])) for k in complete[0]['metrics'][name]} for name in complete[0]['metrics']}
    # Reload checkpoint and verify one complete held-out true-embedding inference.
    reload_model=Decoder().cuda().eval();reload_model.load_state_dict(torch.load(out/'image_decoder.pt',weights_only=True)['state_dict'])
    check=items[index['images'][0]['image_id']]['x']
    with torch.no_grad():reload_error=float((model(check)-reload_model(check)).abs().max())
    assert reload_error==0.
    columns=['original','true_embedding','best_kv_inverse','prefix_rollout']
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:
        font=ImageFont.load_default(size=17)
    def panel(which,path):
        w,h=280,280;canvas=Image.new('RGB',(90+4*w,55+len(which)*(h+25)),'white');draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(['Original (downsampled)','True embedding','KV inverse (best)','Prefix net rollout']):draw.text((95+j*w,15),label,font=font,fill='black')
        for i,(iid,arrays,full) in enumerate(which):
            y=55+i*(h+25);draw.text((5,y+20),iid+('\npartial' if not full else ''),font=font,fill='black')
            for j,name in enumerate(columns):
                im=Image.fromarray(np.round(arrays[name]*255).astype(np.uint8));im.thumbnail((w-10,h),Image.Resampling.LANCZOS)
                canvas.paste(im,(95+j*w,y))
        canvas.save(path)
    panel(panels,out/'all_examples.png');panel(panels[:4],out/'overview.png')
    report=dict(selected_decoder=selected,cnn_validation_mse=best,ridge_validation_mse=ridge_best,
                training_seconds=time.perf_counter()-start,cnn_best_step=best_step,train_images=40,val_images=8,test_images=8,
                metrics_7_complete_images=aggregate,per_image=rows,checkpoint_reload_max_abs=reload_error,
                limits=['Pixel decoder trained only on 40 real-image embedding/target pairs; model capacity/data limit conclusions.',
                        'Quarter native-input resolution (12 output pixels per 48-pixel token cell); no full-resolution claim.',
                        'Ground-truth reference uses PIL Lanczos resizing, not byte-identical phone STB preprocessing.',
                        'Token grids reconstructed in row-major spatial order from known resized dimensions.',
                        'Last test image has missing observations; mean embeddings fill missing input and those pixels are masked gray.',
                        'Aggregate metrics exclude partial final image; its per-image metrics use observed cells only.',
                        'Primary inverse is validation-selected KV ridge; prefix rollout uses separately reported last-step trained model.',
                        'Test images and pixels never used to train or select either model. No generated replacement images.',
                        'No NPU re-forward of predicted embeddings and no claim of exact image recovery.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    emit('RECONSTRUCTION_COMPLETE',report)


if __name__=='__main__':main()
