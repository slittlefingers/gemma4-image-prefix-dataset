"""Fixed-protocol real-NPU dataset: (strict prefix, current INT8 KV, position) -> E.

No target-model forward or gradient is used. Errors here are embedding errors,
not errors of re-running the predicted embedding on the phone.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]/'data/20261022_pilot_prefill'
BINS = [(10,32),(32,64),(64,128),(128,256),(256,512),(512,1000),(1000,1500),(1500,2001)]


def emit(name, obj):
    print(name, json.dumps(obj, ensure_ascii=True), flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_data(root, device):
    index = json.loads((root/'index.json').read_text())
    manifest = json.loads((root/'image_manifest.json').read_text())
    images = {}
    split_ids = {s:set() for s in ['train','validation','test']}
    hashes = {s:set() for s in split_ids}
    for item in manifest:
        path = root/'images'/f"{item['id']}.npz"
        assert digest(path) == item['feature_sha256'], path
        with np.load(path, allow_pickle=False) as f:
            images[int(item['id'])] = f['embedding'].copy()
            assert all(np.isfinite(f[k]).all() for k in f.files)
        split_ids[item['split']].add(int(item['id']))
        hashes[item['split']].add(item['source_sha256'])
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not split_ids[a]&split_ids[b]
        assert not hashes[a]&hashes[b]
    sequences = []
    for item in index:
        path = root/item['file']
        assert digest(path) == item['sha256'], path
        with np.load(path, allow_pickle=False) as f:
            d = {k:torch.from_numpy(f[k].copy()).to(device) for k in f.files}
        e = d['embeddings']; p = d['target_positions'].long()
        assert e.shape == (2001,1536) and d['kv_codes'].shape == (2001,1024)
        assert d['kv_codes'].dtype == torch.int8 and torch.isfinite(e).all()
        assert p.min() == 10 and p.max() == 2000 and len(p) == 1977
        for image_id in d['image_ids'][p].unique().tolist():
            assert image_id in split_ids[item['split']]
            mask = d['image_ids'] == image_id
            actual = e[mask].cpu().numpy()
            expected = images[image_id][d['image_rows'][mask].cpu().numpy()]
            assert np.array_equal(actual,expected)
        d.update(id=item['id'],split=item['split'])
        sequences.append(d)
    audit = dict(sequence_hashes='all match', image_feature_hashes='all match',
                 image_and_source_hash_split_overlap=0, image_labels='bitwise equal',
                 counts={s:dict(sequences=sum(d['split']==s for d in sequences),
                      images=len(split_ids[s]),targets=sum(len(d['target_positions']) for d in sequences if d['split']==s)) for s in split_ids},
                 quantization=json.loads((root/'quantization.json').read_text()),
                 note='Original images not in archive; labels and observations are real phone-collected arrays.')
    return sequences, audit


def metrics(pred, target):
    error = (pred.double()-target.double())
    mse = float(error.square().mean())
    signal = float(target.double().square().mean())
    return dict(rmse=math.sqrt(mse), relative_rmse=math.sqrt(mse/signal),
                mse=mse, target_rms=math.sqrt(signal),
                mean_token_l2=float(error.norm(dim=-1).mean()),
                mean_cosine=float(F.cosine_similarity(pred.double(),target.double(),dim=-1).mean()))


def features(kv, kind):
    return kv[..., :512] if kind == 'k' else kv[...,512:] if kind == 'v' else kv


def predict_ridge(kv, state):
    x=(features(kv.float(),state['kind'])-state['xm'])/state['xs']
    return (x@state['weight']+state['bias'])*state['ys']+state['ym']


def fit_ridges(data, out):
    train=[d for d in data if d['split']=='train']; val=next(d for d in data if d['split']=='validation')
    kv=torch.cat([d['kv_codes'][d['target_positions']].float() for d in train])
    y=torch.cat([d['embeddings'][d['target_positions']] for d in train])
    vp=val['target_positions']; vy=val['embeddings'][vp]
    ym=y.mean(0); ys=y.std(0).clamp_min(.05)
    yn=(y-ym)/ys
    states={}; reports={}
    for kind in ['k','v','kv']:
        xx=features(kv,kind); xm=xx.mean(0); xs=xx.std(0).clamp_min(1)
        x=(xx-xm)/xs
        # Mean-normalized ridge: alpha multiplies identity after division by N.
        xd=x.double(); yd=yn.double()
        gram=xd.T@xd/len(x); rhs=xd.T@yd/len(x)
        ev,vec=torch.linalg.eigh(gram)
        rotated=vec.T@rhs
        records=[]; best=float('inf')
        for alpha in [.0001,.001,.01,.1,1.]:
            w=(vec@(rotated/(ev[:,None]+alpha))).float()
            state=dict(kind=kind,xm=xm,xs=xs,ym=ym,ys=ys,weight=w,bias=yn.mean(0))
            m=metrics(predict_ridge(val['kv_codes'][vp],state),vy)
            records.append(dict(alpha=alpha,validation=m))
            if m['mse'] < best:
                best=m['mse']; states[kind]=state; selected=alpha
        reports[kind]=dict(selected_alpha=selected,grid=records)
        emit('RIDGE',dict(kind=kind,alpha=selected,validation_mse=best))
    torch.save({k:{a:b.cpu() if torch.is_tensor(b) else b for a,b in s.items()} for k,s in states.items()},out/'ridges.pt')
    save_json(out/'ridge_validation.json',reports)
    return states, reports, ym, ys


def position_encoding(n, width, device):
    t=torch.arange(n,device=device).float()[:,None]
    freq=torch.exp(torch.arange(0,width,2,device=device).float()*(-math.log(10000)/width))
    return torch.stack((torch.sin(t*freq),torch.cos(t*freq)),dim=-1).flatten(1)


class Block(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.05),nn.Linear(width*2,width))

    def forward(self,h,memory,mask):
        h=h+self.attn(self.norm(h),memory,memory,attn_mask=mask,need_weights=False)[0]
        return h+self.ff(h)


class Inverse(nn.Module):
    def __init__(self, ridge, mode='prefix', width=192):
        super().__init__()
        self.mode=mode;self.kind=ridge['kind'];self.width=width
        for name in ['xm','xs','ym','ys','weight','bias']:
            self.register_buffer(name,ridge[name].clone())
        self.register_buffer('pos',position_encoding(2001,width,ridge['ym'].device))
        self.kv_in=nn.Linear(1024,width)
        # KV input normalization is set separately from the optional V-only ridge.
        self.register_buffer('kv_mean',torch.zeros(1024,device=ridge['ym'].device))
        self.register_buffer('kv_std',torch.ones(1024,device=ridge['ym'].device))
        self.memory_in=nn.Sequential(nn.Linear(1536,width),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([Block(width),Block(width)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1536))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def base(self,kv):
        return ((features(kv,self.kind)-self.xm)/self.xs)@self.weight+self.bias

    def memory(self,e,p):
        return self.memory_in((e-self.ym)/self.ys)+self.pos[p]

    def forward(self,kv,p,prefix):
        h=self.kv_in((kv.float()-self.kv_mean)/self.kv_std)+self.pos[p]
        if self.mode=='prefix':
            memory=self.memory(prefix,torch.arange(prefix.shape[1],device=prefix.device))
            # Queries are independent: no attention between current/future KVs.
            # nn.MultiheadAttention boolean True means forbidden.
            mask=torch.arange(prefix.shape[1],device=p.device)[None,:]>=p[:,None]
            for block in self.blocks:
                h=block(h,memory,mask)
        else:
            for block in self.blocks:
                h=h+block.ff(h)
        return (self.base(kv.float())+self.head(h))*self.ys+self.ym

    def project_memory(self,memory,block):
        w=block.attn.in_proj_weight;b=block.attn.in_proj_bias;d=self.width
        return [F.linear(memory,w[j*d:(j+1)*d],b[j*d:(j+1)*d]).reshape(1,-1,4,d//4).transpose(1,2) for j in [1,2]]

    @torch.no_grad()
    def rollout(self,d,true_history=False,return_full=False):
        # Recover every image row, including first five rows outside target_positions.
        # Initial template and EOI/BOI rows are known protocol constants.
        e=d['embeddings'];kv=d['kv_codes'].float();n=len(e)
        history=e[:5].clone(); memory=self.memory(history,torch.arange(5,device=e.device))[None]
        cache=[]
        for block in self.blocks:
            k,v=self.project_memory(memory,block)
            kk=torch.zeros(1,4,n,self.width//4,device=e.device);vv=torch.zeros_like(kk)
            kk[:,:,:5]=k;vv[:,:,:5]=v;cache.append((kk,vv))
        output=torch.empty_like(e);output[:5]=e[:5]
        for p in range(5,n):
            if d['token_types'][p].item()==1:
                h=self.kv_in((kv[p:p+1]-self.kv_mean)/self.kv_std)[None]+self.pos[p]
                for block,(k,v) in zip(self.blocks,cache):
                    if self.mode=='prefix':
                        q=F.linear(block.norm(h),block.attn.in_proj_weight[:self.width],block.attn.in_proj_bias[:self.width])
                        q=q.reshape(1,1,4,self.width//4).transpose(1,2)
                        z=F.scaled_dot_product_attention(q,k[:,:,:p],v[:,:,:p]).transpose(1,2).reshape(1,1,self.width)
                        h=h+block.attn.out_proj(z)
                    h=h+block.ff(h)
                predicted=((self.base(kv[p:p+1])+self.head(h)[0])*self.ys+self.ym)[0]
                output[p]=predicted
            else:
                output[p]=e[p]
            if self.mode=='prefix':
                next_e=e[p] if true_history else output[p]
                memory=self.memory(next_e[None],torch.tensor([p],device=e.device))[None]
                for block,(k,v) in zip(self.blocks,cache):
                    a,b=self.project_memory(memory,block);k[:,:,p:p+1]=a;v[:,:,p:p+1]=b
        return output if return_full else output[d['target_positions']]


@torch.no_grad()
def predict(model,d,prefix=None,chunk=256):
    e=d['embeddings'] if prefix is None else prefix
    output=[]
    for p in d['target_positions'].split(chunk):
        output.append(model(d['kv_codes'][p][None],p,e[None])[0])
    return torch.cat(output)


def train_model(data,ridge,mode,args,out,noise=0.):
    torch.manual_seed(args.seed)
    model=Inverse(ridge,mode).cuda()
    train=[d for d in data if d['split']=='train'];val=next(d for d in data if d['split']=='validation')
    trainkv=torch.cat([d['kv_codes'][d['target_positions']].float() for d in train])
    model.kv_mean.copy_(trainkv.mean(0));model.kv_std.copy_(trainkv.std(0).clamp_min(1))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    # Equal samples from eight prefix-length bins. One sequence per step shares projection.
    bucket=[[d['target_positions'][(d['target_positions']>=lo)&(d['target_positions']<hi)] for lo,hi in BINS] for d in train]
    scale=torch.cat([d['embeddings'][d['target_positions']] for d in train]).square().mean()
    best=math.inf;trace=[];start=time.perf_counter();recent_losses=[]
    initial_memory=model.memory_in[0].weight.detach().clone()
    for step in range(args.steps+1):
        if step%100==0 or step==args.steps:
            model.eval(); pred=predict(model,val);m=metrics(pred,val['embeddings'][val['target_positions']])
            row=dict(step=step,validation=m,seconds=time.perf_counter()-start,
                     recent_train_nmse=float(np.mean(recent_losses)) if recent_losses else None)
            recent_losses=[]
            trace.append(row);emit('VALIDATION',dict(mode=mode,noise=noise,**row))
            if m['mse']<best:
                best=m['mse'];best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if step==args.steps:break
        model.train();i=int(torch.randint(len(train),()).item());d=train[i]
        p=torch.cat([b[torch.randint(len(b),(8,),device='cuda')] for b in bucket[i]])
        e=d['embeddings'][None]
        if noise:
            # Synthetic embedding noise, not a measured NPU error model.
            e=e+torch.randn_like(e)*model.ys*noise
        opt.zero_grad(set_to_none=True)
        prediction=model(d['kv_codes'][p][None],p,e)[0]
        loss=(prediction-d['embeddings'][p]).square().mean()/scale
        recent_losses.append(float(loss.detach()))
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        lr=3e-4*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.step()
    model.eval()
    final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Fixed training subset for diagnosing fit vs held-out generalization.
    train_errors=[];train_signals=[]
    with torch.no_grad():
        for d in train:
            pp=d['target_positions'][::8]
            yp=model(d['kv_codes'][pp][None],pp,d['embeddings'][None])[0]
            train_errors.append((yp-d['embeddings'][pp]).square().flatten())
            train_signals.append(d['embeddings'][pp].square().flatten())
    final_train_mse=float(torch.cat(train_errors).mean())
    memory_weight_change=float((model.memory_in[0].weight.detach()-initial_memory).abs().max())
    model.load_state_dict(state);model.eval()
    metadata=dict(mode=mode,seed=args.seed,steps=args.steps,best_step=best_step,
                  best_validation_mse=best,batch_queries=64,prefix_length_balanced=True,
                  synthetic_prefix_noise_sigma=noise,parameters=sum(p.numel() for p in model.parameters()),
                  seconds=time.perf_counter()-start,trace=trace,ridge_kind=ridge['kind'],
                  last_step_train_subset_mse=final_train_mse,
                  last_step_train_subset_relative_rmse=math.sqrt(final_train_mse/float(torch.cat(train_signals).mean())),
                  last_step_prefix_projection_weight_change=memory_weight_change,
                  selection='Validation teacher-input embedding MSE only; test never used for selection.')
    torch.save(dict(state_dict=state,metadata=metadata),out/f'{mode}_model.pt')
    torch.save(dict(state_dict=final_state,metadata=metadata,selection='Last training step, not validation-selected'),out/f'{mode}_last.pt')
    save_json(out/f'{mode}_training.json',metadata)
    return model,metadata


@torch.no_grad()
def audit_model(model,d,out,checkpoint_name='prefix_model.pt'):
    p=d['target_positions'][100:104];e=d['embeddings'];kv=d['kv_codes'][p][None]
    first=model(kv,p,e[None])[0]
    errors=[]
    for j,pos in enumerate(p.tolist()):
        altered=e.clone();altered[pos:]=torch.randn_like(altered[pos:])*10
        got=model(kv[:,j:j+1],p[j:j+1],altered[None])[0,0]
        ref=model(kv[:,j:j+1],p[j:j+1],e[None])[0,0]
        errors.append(float((got-ref).abs().max()))
    assert max(errors)==0.,errors
    history=model.rollout(d,true_history=True)
    parallel=predict(model,d)
    difference=float((history-parallel).abs().max())
    assert difference<2e-4,difference
    checkpoint=torch.load(out/checkpoint_name,weights_only=True,map_location='cuda')
    clone=copy.deepcopy(model);clone.load_state_dict(checkpoint['state_dict']);clone.eval()
    reload=float((clone(kv,p,e[None])[0]-first).abs().max());assert reload==0.
    return dict(current_future_embedding_mutation_max_abs=max(errors),
                independent_queries=True,parallel_vs_sequential_exact_history_max_abs=difference,
                checkpoint_reload_max_abs=reload)


def grouped_metrics(pred,target,d):
    p=d['target_positions'];ids=d['image_ids'][p]
    return dict(overall=metrics(pred,target),
                prefix_bins={f'{lo}-{hi-1}':metrics(pred[(p>=lo)&(p<hi)],target[(p>=lo)&(p<hi)]) for lo,hi in BINS},
                images={str(i):dict(count=int((ids==i).sum()),**metrics(pred[ids==i],target[ids==i])) for i in ids.unique().tolist()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,default=HERE/'results');ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20261022);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    data,audit=load_data(args.data,'cuda');save_json(args.out/'data_audit.json',audit);emit('DATA_AUDIT',audit)
    states,ridge_reports,ym,ys=fit_ridges(data,args.out)
    # Choose ridge initialization using validation only, across K/V/KV and alphas.
    kind=min(states,key=lambda k:min(row['validation']['mse'] for row in ridge_reports[k]['grid']))
    emit('RIDGE_INITIALIZATION',kind)
    kv_model,kv_training=train_model(data,states[kind],'kv_only',args,args.out)
    prefix_model,prefix_training=train_model(data,states[kind],'prefix',args,args.out)
    val=next(d for d in data if d['split']=='validation')
    audit['model']=audit_model(prefix_model,val,args.out);save_json(args.out/'data_audit.json',audit)
    last_model=copy.deepcopy(prefix_model)
    last_model.load_state_dict(torch.load(args.out/'prefix_last.pt',weights_only=True,map_location='cuda')['state_dict'])
    last_model.eval()
    audit['last_step_model']=audit_model(last_model,val,args.out,'prefix_last.pt')
    save_json(args.out/'data_audit.json',audit)
    d=next(d for d in data if d['split']=='test');p=d['target_positions'];target=d['embeddings'][p]
    predictions={f'ridge_{kind}':predict_ridge(d['kv_codes'][p],s).detach() for kind,s in states.items()}
    predictions.update(mean=ym[None].expand_as(target),previous_true_embedding=d['embeddings'][p-1])
    with torch.no_grad():
        predictions['kv_only_model']=predict(kv_model,d)
        predictions['prefix_true']=predict(prefix_model,d)
        predictions['prefix_mean']=predict(prefix_model,d,ym[None].expand_as(d['embeddings']))
        predictions['prefix_rollout']=prefix_model.rollout(d)
        # Diagnostic only: last checkpoint was not selected using test results.
        predictions['prefix_last_true']=predict(last_model,d)
        predictions['prefix_last_rollout']=last_model.rollout(d)
        predictions['prefix_last_mean']=predict(last_model,d,ym[None].expand_as(d['embeddings']))
    results={name:grouped_metrics(pred,target,d) for name,pred in predictions.items()}
    torch.save(dict(predictions={k:v.detach().cpu() for k,v in predictions.items()},target=target.cpu(),
                    positions=p.cpu(),image_ids=d['image_ids'][p].cpu()),args.out/'test_predictions.pt')
    report=dict(dataset=str(args.data.resolve()),seed=args.seed,audit=audit,training=dict(kv_only=kv_training,prefix=prefix_training),
                test_sequence=d['id'],results=results,
                limits=['8 held-out images in one test sequence; positions are correlated.',
                        'Embedding errors only. No phone attached; no NPU re-forward KV error measured.',
                        'Training uses exact prefixes. Rollout substitutes all previous image predictions from position 5.',
                        'Known template/separator embeddings and their positions retained in rollout.',
                        'Raw quantized K/V and absolute position are inputs; no guessed inverse RoPE.',
                        'No original image reconstruction training; archive contains embeddings and KV, not JPEGs.',
                        'Dataset fixed-prefill protocol differs measurably from single-position query path.'])
    save_json(args.out/'summary.json',report)
    emit('COMPLETE',dict(results={k:v['overall'] for k,v in results.items()},model_audit=audit['model']))


if __name__=='__main__':main()
