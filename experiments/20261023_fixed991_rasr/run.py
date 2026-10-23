"""Fixed991 pilot: image-ID split, full-image joint reconstruction, held-out evaluation.

Run: python run.py --epochs 12
The independent whiteboard truth is opened only after all checkpoints are frozen
and its KV-only reconstruction has been saved.
"""
import argparse,hashlib,json,math,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
import torch.nn.functional as F
from model import Reconstructor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'data/20261023_fixed991_1000'
OUT=HERE/'results'
SEED=20261023

def emit(tag,data):print(tag,json.dumps(data),flush=True)
def dump(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save_image(x,path):
    if isinstance(x,torch.Tensor):x=x.detach()[0].permute(1,2,0).float().cpu().numpy()
    Image.fromarray(np.uint8(np.clip(x*255+.5,0,255))).save(path)

def metrics(pred,target):
    a=pred.detach()[0].permute(1,2,0).float().cpu().numpy().astype(np.float64)
    b=target.detach()[0].permute(1,2,0).float().cpu().numpy().astype(np.float64)
    mse=float(np.mean((a-b)**2));ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    s=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return dict(mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),ssim=float(s[5:-5,5:-5].mean()))

def load_data():
    entries=json.loads((ROOT/'index.json').read_text())
    ordered=sorted(entries,key=lambda r:hashlib.sha256(('fixed991-pilot-v1:'+r['source_image_id']).encode()).hexdigest())
    splits={r['id']:('train' if i<800 else 'validation' if i<900 else 'test') for i,r in enumerate(ordered)}
    assert len(entries)==1000 and len(set(r['source_image_id'] for r in entries))==1000
    assert len(set(r['source_sha256'] for r in entries))==1000
    target_meta=json.loads((ROOT/'important_test/whiteboard/metadata.json').read_text())
    assert target_meta['source_sha256'] not in {r['source_sha256'] for r in entries}
    def read(r):
        p=ROOT/r['file'];assert digest(p)==r['sha256'],r['id']
        with np.load(p,allow_pickle=False) as f:
            k=f['kv_codes'];e=f['embedding'];pixels=f['pixels'];grid=tuple(int(x) for x in f['grid'])
            assert k.dtype==np.int8 and k.shape==(grid[0]*grid[1],1024)
            assert e.dtype==np.float32 and e.shape==(len(k),1536) and np.isfinite(e).all()
            assert np.isfinite(f['pre_adapter']).all()
            assert pixels.dtype==np.uint8 and pixels.shape==(grid[0]*48,grid[1]*48,3)
            assert np.array_equal(f['positions'],np.arange(991,991+len(k)))
            assert r['image_start']==991 and r['prefix_sha256']==target_meta['prefix_sha256']
            # Exact 2x area reduction from native STB labels; no external JPEG re-resize.
            y=pixels.astype(np.float32).reshape(grid[0]*24,2,grid[1]*24,2,3).mean((1,3))/255
        return dict(id=r['id'],split=splits[r['id']],grid=grid,k=k,e=e,y=y)
    data=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,d in enumerate(pool.map(read,entries),1):
            d['k']=torch.from_numpy(d['k']).cuda();d['e']=torch.from_numpy(d['e']).cuda()
            d['y']=torch.from_numpy(d['y']).permute(2,0,1)[None].half().cuda();data.append(d)
            if i%100==0:emit('LOAD',dict(images=i))
    audit=dict(images=len(data),split_counts={s:sum(d['split']==s for d in data) for s in ['train','validation','test']},
               split_ids={s:[d['id'] for d in data if d['split']==s] for s in ['train','validation','test']},
               all_npz_sha256_verified=True,source_id_hash_disjoint=True,target_source_hash_disjoint=True,
               prefix_tokens=991,layer_index=4,raw_kv_dtype='int8',embedding_dtype='float32',output_pixels_per_token=24,
               pixel_target='2x2 area reduction of native STB RGB uint8; half native width and height',seed=SEED)
    dump(OUT/'data_audit.json',audit)
    return data,audit

def moments(data,key,minimum):
    width=data[0][key].shape[-1];total=torch.zeros(width,device='cuda',dtype=torch.float64);sq=total.clone();n=0
    for d in data:
        x=d[key].double();total+=x.sum(0);sq+=(x*x).sum(0);n+=len(x)
    mean=total/n;std=((sq-total*total/n)/(n-1)).clamp_min(0).sqrt().clamp_min(minimum)
    return mean.float(),std.float()

def fit_ridge(train,val,source,target,mean,std,name):
    dim=len(mean);gram=torch.zeros(dim+1,dim+1,device='cuda',dtype=torch.float64)
    odim=1536 if target=='e' else 3*24*24;rhs=torch.zeros(dim+1,odim,device='cuda',dtype=torch.float64);n=0
    for d in train:
        x=((d[source].float()-mean)/std).double();x=torch.cat([x,torch.ones(len(x),1,device='cuda',dtype=torch.float64)],1)
        y=d['e'].double() if target=='e' else F.unfold(d['y'].float(),24,stride=24)[0].T.double()
        gram+=x.T@x;rhs+=x.T@y;n+=len(x)
    gram/=n;rhs/=n;xm=gram[:-1,-1];ym=rhs[-1]
    eigen,vec=torch.linalg.eigh(gram[:-1,:-1]-xm[:,None]*xm[None,:]);proj=vec.T@(rhs[:-1]-xm[:,None]*ym[None,:])
    best=math.inf;trace=[]
    for alpha in [.0001,.001,.01,.1,1.]:
        w=(vec@(proj/(eigen[:,None]+alpha))).float();b=(ym-xm@w.double()).float();errors=[]
        for d in val:
            p=((d[source].float()-mean)/std)@w+b
            y=d['e'] if target=='e' else F.unfold(d['y'].float(),24,stride=24)[0].T
            if target!='e':p=p.clamp(0,1)
            errors.append(float(F.mse_loss(p,y)))
        loss=float(np.mean(errors));trace.append(dict(alpha=alpha,validation_mse=loss))
        if loss<best:best=loss;chosen=dict(w=w.clone(),b=b.clone(),mean=mean,std=std,alpha=alpha)
    emit('RIDGE',dict(name=name,trace=trace));return chosen,trace

def ridge_apply(x,r):return ((x.float()-r['mean'])/r['std'])@r['w']+r['b']
def patch_image(rows,grid):return F.fold(rows.T[None],(grid[0]*24,grid[1]*24),24,stride=24).clamp(0,1)

@torch.no_grad()
def neural_predict(model,d,mean,std):
    x=(d['k'].float()-mean)/std
    with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x[None],d['grid'])
    return p.float()

def train_joint(train,val,mean,std,epochs):
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    model=Reconstructor().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    groups=defaultdict(list)
    for i,d in enumerate(train):groups[d['grid']].append(i)
    steps_per_epoch=sum(math.ceil(len(ids)/3) for ids in groups.values());total=epochs*steps_per_epoch
    emit('MODEL',dict(parameters=sum(p.numel() for p in model.parameters()),epochs=epochs,batch_size=3,steps=total,
                       lr=3e-4,warmup_steps=100,schedule='linear',pixel_loss='MSE only',intermediate_supervision=False))
    best=math.inf;step=0;trace=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        batches=[]
        for ids in groups.values():
            ids=rng.permutation(ids).tolist();batches.extend(ids[i:i+3] for i in range(0,len(ids),3))
        rng.shuffle(batches);model.train();loss_sum=0;n=0
        for ids in batches:
            batch=[train[i] for i in ids];x=torch.stack([(d['k'].float()-mean)/std for d in batch]);y=torch.cat([d['y'] for d in batch]).float()
            step+=1;lr=3e-4*min(step/100,1)*max(0,(total-step)/max(total-100,1))
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,batch[0]['grid'])
            loss=F.mse_loss(pred.float(),y);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum+=float(loss.detach())*len(ids);n+=len(ids)
            if step==1 or step%100==0:emit('TRAIN',dict(step=step,epoch=epoch,loss=float(loss.detach()),seconds=time.perf_counter()-start))
        model.eval()
        with torch.no_grad():vm=float(np.mean([float(F.mse_loss(neural_predict(model,d,mean,std),d['y'].float())) for d in val]))
        if vm<best:
            best=vm;best_epoch=epoch
            checkpoint=dict(state_dict=cpu_state(model),kv_mean=mean.cpu(),kv_std=std.cpu(),epoch=epoch,validation_mse=best,
                            input='raw signed INT8 K then V, each 512 dims',output_pixels_per_token=24,
                            architecture='1 causal Transformer(256), latent1536, MLP->768, inverse vision Transformer192, U-net-style conv and pixelshuffle',
                            train_ids=[d['id'] for d in train],validation_ids=[d['id'] for d in val])
            torch.save(checkpoint,OUT/'joint_best.pt')
        row=dict(epoch=epoch,step=step,train_mse=loss_sum/n,validation_mse=vm,best_epoch=best_epoch,seconds=time.perf_counter()-start)
        trace.append(row);dump(OUT/'training.json',trace);emit('EPOCH',row)
    torch.save(dict(state_dict=cpu_state(model),epoch=epochs),OUT/'joint_last.pt')
    checkpoint=torch.load(OUT/'joint_best.pt',weights_only=True,map_location='cpu');model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,trace

def panel(rows,path,cols):
    font=ImageFont.load_default(size=15);cw=310;rh=275
    canvas=Image.new('RGB',(cw*len(cols),40+rh*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    for j,(key,label) in enumerate(cols):draw.text((j*cw+6,8),label,font=font,fill='black')
    for i,(name,arrays) in enumerate(rows):
        for j,(key,_) in enumerate(cols):
            im=Image.fromarray(np.uint8(np.clip(arrays[key].detach()[0].permute(1,2,0).float().cpu().numpy()*255+.5,0,255)))
            im.thumbnail((cw-12,rh-28),Image.Resampling.LANCZOS);x=j*cw+6;y=40+i*rh
            canvas.paste(im,(x,y));draw.text((x,y+rh-22),name[-20:],font=font,fill='black')
    canvas.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--reuse',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    data,audit=load_data();train=[d for d in data if d['split']=='train'];val=[d for d in data if d['split']=='validation'];test=sorted([d for d in data if d['split']=='test'],key=lambda d:d['id'])
    km,ks=moments(train,'k',1.);em,es=moments(train,'e',.05)
    if args.reuse:
        pack=torch.load(OUT/'ridges.pt',weights_only=True,map_location='cuda');rke,rep,rkp=[pack[k] for k in ['kv_e','e_pixels','kv_pixels']];ridge_trace=pack['trace']
        ck=torch.load(OUT/'joint_best.pt',weights_only=True,map_location='cuda');model=Reconstructor().cuda();model.load_state_dict(ck['state_dict']);model.eval();trace=json.loads((OUT/'training.json').read_text())
        assert torch.equal(km,ck['kv_mean']) and torch.equal(ks,ck['kv_std'])
    else:
        rke,tke=fit_ridge(train,val,'k','e',km,ks,'KV -> E')
        rep,tep=fit_ridge(train,val,'e','pixels',em,es,'true E -> pixels')
        rkp,tkp=fit_ridge(train,val,'k','pixels',km,ks,'KV -> pixels')
        ridge_trace=dict(kv_e=tke,e_pixels=tep,kv_pixels=tkp)
        torch.save(dict(kv_e={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in rke.items()},
                        e_pixels={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in rep.items()},
                        kv_pixels={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in rkp.items()},trace=ridge_trace),OUT/'ridges.pt')
        model,trace=train_joint(train,val,km,ks,args.epochs)
    frozen=dict(joint_checkpoint_sha256=digest(OUT/'joint_best.pt'),ridge_checkpoint_sha256=digest(OUT/'ridges.pt'),
                target_truth_used_for_training_or_selection=False,checkpoint_selection='minimum mean per-image validation pixel MSE',
                target_inference_inputs=['observed.npz: codes, grid','training-set KV normalization'])
    dump(OUT/'frozen_before_test.json',frozen)
    rows=[];panels=[]
    @torch.no_grad()
    def predict(d):
        ehat=ridge_apply(d['k'],rke)
        arrays=dict(joint=neural_predict(model,d,km,ks),separate_chain=patch_image(ridge_apply(ehat,rep),d['grid']),
                    direct_ridge=patch_image(ridge_apply(d['k'],rkp),d['grid']))
        return arrays,ehat
    with torch.no_grad():
        for i,d in enumerate(test):
            arrays,ehat=predict(d);arrays['true_e']=patch_image(ridge_apply(d['e'],rep),d['grid']);arrays['original']=d['y'].float()
            dm=dict(d);dm['k']=km.expand(len(d['k']),-1);arrays['mean_kv']=neural_predict(model,dm,km,ks)
            ds=dict(d);ds['k']=d['k'][torch.randperm(len(d['k']),device='cuda')];arrays['shuffled_kv']=neural_predict(model,ds,km,ks)
            row=dict(id=d['id'],grid=d['grid'],metrics={k:metrics(p,d['y'].float()) for k,p in arrays.items() if k!='original'},
                     embedding_rmse=float((ehat-d['e']).square().mean().sqrt()))
            rows.append(row)
            if i<6:
                panels.append((d['id'],{k:p.clone() for k,p in arrays.items()}))
                for k,p in arrays.items():save_image(p,OUT/f"test_{i:03d}_{k}.png")
        target_dir=ROOT/'important_test/whiteboard';meta=json.loads((target_dir/'metadata.json').read_text())
        assert digest(target_dir/'observed.npz')==meta['observation_sha256']
        with np.load(target_dir/'observed.npz',allow_pickle=False) as f:
            d=dict(k=torch.from_numpy(f['codes'].copy()).cuda(),grid=tuple(int(v) for v in f['grid']))
            assert np.array_equal(f['positions'],np.arange(991,991+len(d['k'])))
        arrays,ehat=predict(d)
        for k,p in arrays.items():save_image(p,OUT/f'whiteboard_{k}.png')
        torch.save({k:p.cpu() for k,p in arrays.items()},OUT/'whiteboard_kv_only_predictions.pt')
        # First access to whiteboard truth inside this experiment: evaluation only.
        with np.load(target_dir/'evaluation_only_truth.npz',allow_pickle=False) as f:
            e=torch.from_numpy(f['embedding'].copy()).cuda();native=torch.from_numpy(f['pixels'].copy()).cuda().permute(2,0,1)[None].float()/255
            y=F.avg_pool2d(native,2)
        arrays['true_e']=patch_image(ridge_apply(e,rep),d['grid']);arrays['original']=y
        wm={k:metrics(p,y) for k,p in arrays.items() if k!='original'}
        for k,p in arrays.items():save_image(p,OUT/f'whiteboard_{k}.png')
        save_image(native,OUT/'whiteboard_native_original.png')
        cols=[('original','Native pixels / 2'),('true_e','True E -> pixels'),('separate_chain','KV -> E -> pixels'),('direct_ridge','KV -> pixels (ridge)'),('joint','KV -> joint Transformer')]
        panel([('independent whiteboard',arrays)],OUT/'whiteboard_comparison.png',cols)
        panel(panels,OUT/'test_overview.png',cols)
        # Roundtrip checkpoint and batch-size audit; only uses validation observations.
        ck=torch.load(OUT/'joint_best.pt',weights_only=True,map_location='cpu');reloaded=Reconstructor().cuda();reloaded.load_state_dict(ck['state_dict']);reloaded.eval()
        p1=neural_predict(model,val[0],km,ks);p2=neural_predict(reloaded,val[0],km,ks)
        reload_delta=float((p1-p2).abs().max());assert reload_delta==0
    aggregate={method:{metric:float(np.mean([r['metrics'][method][metric] for r in rows])) for metric in ['mse','psnr_db','ssim']} for method in rows[0]['metrics']}
    rng=np.random.default_rng(42);boot={}
    for baseline in ['separate_chain','direct_ridge']:
        differences=np.array([r['metrics']['joint']['mse']-r['metrics'][baseline]['mse'] for r in rows]);samples=differences[rng.integers(len(rows),size=(10000,len(rows)))].mean(1)
        boot[baseline]=dict(mean_mse_difference=float(differences.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
    summary=dict(dataset='fixed991_1000',dataset_commit='955a392',splits=audit['split_counts'],training=trace,seed=SEED,
                 test_metrics=aggregate,test_per_image=rows,paired_image_bootstrap_joint_minus_baseline=boot,whiteboard_metrics=wm,
                 checkpoint=frozen,reload_max_abs=reload_delta,ridge_validation=ridge_trace,
                 limitations=['Adaptation of the coarse visual branch, not exact RASR reproduction.',
                              'Input is layer-index-4 NPU INT8 K/V, not full hidden states.',
                              'Smaller architecture, 800 training images, changed training schedule; one random seed.',
                              'Outputs have half native width/height (24 pixels per image token).',
                              'No matched differentiable NPU forward: consistency-guided refinement not performed; no regenerated K/V error measured.',
                              'Only one fixed 991-token prefix; no cross-prefix generalization claim.',
                              'No independent target text transcription: no password or exact text recovery accuracy claimed.'])
    dump(OUT/'summary.json',summary);emit('COMPLETE',dict(test=aggregate,whiteboard=wm,bootstrap=boot))

if __name__=='__main__':main()
