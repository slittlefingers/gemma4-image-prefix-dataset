import json,zipfile,random,hashlib,urllib.request,time,concurrent.futures
from pathlib import Path
workspace=Path(__file__).resolve().parents[4]
root=workspace/'kvidence/datasets/shadow_chat'
old=json.loads((root/'image_chat_100/manifest.json').read_text())
out=root/'image_prefix_1000';(out/'media').mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(root/'source/coco2017/annotations_trainval2017.zip') as z:a=json.loads(z.read('annotations/instances_val2017.json'))
ims={i['id']:i for i in a['images']};groups={i:[] for i in ims}
for x in a['annotations']:groups[x['image_id']].append(x)
licenses={x['id']:x for x in a['licenses']}
used={x['coco_image_id'] for x in old['samples']}
eligible=sorted(i for i in ims if i not in used and len(groups[i])>=3 and len({x['category_id'] for x in groups[i]})>=2)
chosen=random.Random(20261023).sample(eligible,900)
# Preserve all original image split assignments, including unused original images.
# Original split was NumPy seed 20261022; read the saved mapping instead.
proto=json.loads((Path(__file__).resolve().parent/'data/20261022_pilot_prefill/protocol.json').read_text())
records=[]
for x in old['samples']:
 x=x.copy();x['split']=proto['split_by_image_id'][x['id']];x['media']=str((root/'image_chat_100'/x['media']).resolve());records.append(x)
def fetch(pair):
 n,i=pair;im=ims[i];p=out/'media'/im['file_name'];url='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+im['file_name']
 if not p.exists():
  for attempt in range(5):
   try:
    data=urllib.request.urlopen(url,timeout=90).read();assert data[:2]==b'\xff\xd8';p.write_bytes(data);break
   except Exception:
    if attempt==4:raise
    time.sleep(attempt+1)
 split='train' if n<720 else ('validation' if n<810 else 'test')
 return dict(id=f'{101+n:04d}',media=str(p.resolve()),media_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),coco_image_id=i,source_url=url,license=licenses[im['license']],split=split)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for n,x in enumerate(pool.map(fetch,enumerate(chosen)),1):
  records.append(x)
  if n%100==0:print('DOWNLOADED',n,flush=True)
old.update(dataset_name='image_prefix_1000',samples=records,sample_count=1000,selection=dict(seed=20261023,additional_images=900,original_split_preserved=True))
(out/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
print('READY',out,flush=True)
