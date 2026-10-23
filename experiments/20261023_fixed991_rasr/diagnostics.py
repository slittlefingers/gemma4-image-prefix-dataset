"""Post-training diagnostics: validation-selected smoothing and edge preservation.

Added after inspecting the first reconstruction. No target-specific tuning,
no change to the neural checkpoint. Bootstrap does not include seed variance.
"""
import json,math
import cv2,numpy as np,torch
from run import HERE,ROOT,OUT,load_data,ridge_apply,patch_image,neural_predict,metrics,panel,save_image,dump
from model import Reconstructor

def blur(x,sigma):
    if sigma==0:return x
    a=x[0].permute(1,2,0).float().cpu().numpy()
    b=cv2.GaussianBlur(a,(0,0),sigma)
    return torch.from_numpy(b).permute(2,0,1)[None].to(x.device)

def edges(x,y):
    def grad(t):
        a=t[0].permute(1,2,0).float().cpu().numpy();a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);a=cv2.GaussianBlur(a,(0,0),1.)
        return np.sqrt(cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)**2+cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)**2).flatten().astype(np.float64)
    a,b=grad(x),grad(y)
    correlation=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else 0.
    return dict(gradient_magnitude_correlation=correlation,gradient_rms_ratio=float(np.sqrt(np.mean(a*a)/max(np.mean(b*b),1e-20))))

@torch.no_grad()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    data,_=load_data();val=[d for d in data if d['split']=='validation'];test=sorted([d for d in data if d['split']=='test'],key=lambda d:d['id'])
    pack=torch.load(OUT/'ridges.pt',weights_only=True,map_location='cuda');rke,rep,rkp=[pack[k] for k in ['kv_e','e_pixels','kv_pixels']]
    ck=torch.load(OUT/'joint_best.pt',weights_only=True,map_location='cuda');m=Reconstructor().cuda();m.load_state_dict(ck['state_dict']);m.eval()
    mean_rgb=torch.stack([d['y'].float().mean((0,2,3)) for d in data if d['split']=='train']).mean(0)
    def simple(d):
        e=ridge_apply(d['k'],rke)
        return dict(separate_chain=patch_image(ridge_apply(e,rep),d['grid']),direct_ridge=patch_image(ridge_apply(d['k'],rkp),d['grid']))
    sigmas=[0,1,2,4,8,12,24];scores={k:{str(s):[] for s in sigmas} for k in ['separate_chain','direct_ridge']}
    for d in val:
        for name,x in simple(d).items():
            for sigma in sigmas:scores[name][str(sigma)].append(float((blur(x,sigma)-d['y'].float()).square().mean()))
    trace={k:{s:float(np.mean(v)) for s,v in choices.items()} for k,choices in scores.items()}
    chosen={k:float(min(choices,key=choices.get)) for k,choices in trace.items()}
    print('SMOOTHING_SELECTED',json.dumps(dict(trace=trace,chosen=chosen)),flush=True)
    def all_predictions(d):
        p=simple(d)
        for name in list(p):p['smooth_'+name]=blur(p[name],chosen[name])
        p['joint']=neural_predict(m,d,ck['kv_mean'],ck['kv_std'])
        p['train_mean_rgb']=mean_rgb[None,:,None,None].expand(1,3,d['grid'][0]*24,d['grid'][1]*24)
        if 'e' in d:p['true_e']=patch_image(ridge_apply(d['e'],rep),d['grid'])
        return p
    rows=[];panels=[];precision=[]
    for i,d in enumerate(test):
        # Evaluate against float32 native pixel reductions, independently of cached FP16 labels.
        with np.load(ROOT/'samples'/f"{d['id']}.npz",allow_pickle=False) as f:
            native=torch.from_numpy(f['pixels'].copy()).cuda().permute(2,0,1)[None].float()/255
            y=torch.nn.functional.avg_pool2d(native,2)
        precision.append(float((y-d['y'].float()).square().mean()))
        arrays=all_predictions(d);result={k:dict(**metrics(p,y),**edges(p,y)) for k,p in arrays.items()}
        rows.append(dict(id=d['id'],metrics=result))
        if i<4:arrays['original']=y;panels.append((d['id'],arrays))
    target=ROOT/'important_test/whiteboard'
    with np.load(target/'observed.npz',allow_pickle=False) as f:d=dict(k=torch.from_numpy(f['codes'].copy()).cuda(),grid=tuple(int(v) for v in f['grid']))
    arrays=all_predictions(d)
    with np.load(target/'evaluation_only_truth.npz',allow_pickle=False) as f:
        y=torch.nn.functional.avg_pool2d(torch.from_numpy(f['pixels'].copy()).cuda().permute(2,0,1)[None].float()/255,2)
        e=torch.from_numpy(f['embedding'].copy()).cuda()
    arrays['true_e']=patch_image(ridge_apply(e,rep),d['grid'])
    target_metrics={k:dict(**metrics(p,y),**edges(p,y)) for k,p in arrays.items()}
    for k,p in arrays.items():save_image(p,OUT/f'whiteboard_diagnostic_{k}.png')
    arrays['original']=y
    cols=[('original','Reference / 2'),('separate_chain','Separate chain'),('smooth_separate_chain','Smoothed separate chain'),('joint','Joint Transformer'),('true_e','True E reference')]
    panel([('whiteboard',arrays)],OUT/'whiteboard_smoothing_control.png',cols)
    panel(panels,OUT/'test_smoothing_control.png',cols)
    aggregate={k:{metric:float(np.mean([r['metrics'][k][metric] for r in rows])) for metric in rows[0]['metrics'][k]} for k in rows[0]['metrics']}
    rng=np.random.default_rng(42);boot={}
    for baseline in ['smooth_separate_chain','smooth_direct_ridge']:
        diff=np.array([r['metrics']['joint']['mse']-r['metrics'][baseline]['mse'] for r in rows]);bs=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
        boot[baseline]=dict(mean_mse_difference=float(diff.mean()),ci95=np.quantile(bs,[.025,.975]).tolist())
    result=dict(status='post-training diagnostic, no checkpoint changes',smoothing_selection='100 validation images only',
                smoothing_sigma_output_pixels=chosen,validation_trace=trace,test_metrics=aggregate,whiteboard_metrics=target_metrics,
                bootstrap_joint_minus_smoothed=boot,per_image=rows,cached_fp16_pixel_label_mean_mse=float(np.mean(precision)),
                evaluation_labels='float32 2x2 native pixel area reduction',
                edge_metric='Pearson correlation of Sobel gradient magnitudes after Gaussian sigma=1; not a recognition score')
    dump(OUT/'diagnostics.json',result);print('DIAGNOSTICS_COMPLETE',json.dumps({k:result[k] for k in ['test_metrics','whiteboard_metrics','bootstrap_joint_minus_smoothed','cached_fp16_pixel_label_mean_mse']}),flush=True)

if __name__=='__main__':main()
