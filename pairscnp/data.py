from pathlib import Path
import json,hashlib,random,subprocess,os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .paths import ROOT
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):return (x/255.-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
def seed_worker(i):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
class TrainingData(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with np.load(ROOT/r['cache']) as z:x=torch.from_numpy(z['image'].copy()).permute(2,0,1).float();y=torch.from_numpy(z['mask'].astype('int64'));g=torch.from_numpy(z['geometry'].astype('float32'))
        h,w=y.shape
        if min(h,w)<512:
            scale=512/min(h,w);size=(round(h*scale),round(w*scale));x=F.interpolate(x[None],size=size,mode='bilinear',align_corners=False)[0];y=F.interpolate(y[None,None].float(),size=size,mode='nearest')[0,0].long();g=F.interpolate(g[None,None],size=size,mode='nearest')[0,0]
        h,w=y.shape;t=random.randrange(h-512+1);l=random.randrange(w-512+1)
        x=x[:,t:t+512,l:l+512];y=y[t:t+512,l:l+512];g=g[t:t+512,l:l+512]
        for dim in [-1,-2]:
            if random.random()<.5:x=x.flip(dim);y=y.flip(dim);g=g.flip(dim)
        k=random.randrange(4);x=x.rot90(k,(-2,-1));y=y.rot90(k,(-2,-1));g=g.rot90(k,(-2,-1))
        x=(x*(.85+.3*random.random())+255*(-.05+.1*random.random())).clamp(0,255)
        return norm(x),y,g[None],r['id']
@torch.no_grad()
def predict(model,image):
    w,h=image.size;scale=min(1.,1024/max(w,h));image2=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
    x=norm(torch.from_numpy(np.asarray(image2).copy()).permute(2,0,1).float());hh,ww=x.shape[-2:]
    x=F.pad(x,(0,max(0,512-ww),0,max(0,512-hh)),mode='replicate');H,W=x.shape[-2:]
    starts=lambda n:sorted(set(list(range(0,max(1,n-512+1),384))+[n-512]))
    total=torch.zeros(H,W,device='cuda');counts=torch.zeros_like(total)
    for t in starts(H):
        for l in starts(W):
            z=model(x[:,t:t+512,l:l+512][None].cuda());total[t:t+512,l:l+512]+=z.softmax(1)[0,1];counts[t:t+512,l:l+512]+=1
    p=(total/counts)[:hh,:ww];return F.interpolate(p[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
def evaluate(model,rows,out=None):
    from .metrics import metrics
    model.eval();records=[];preds={};shapes={}
    for i,r in enumerate(rows):
        assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['mask'])==r['mask_sha256']
        x=Image.open(ROOT/r['image']).convert('RGB');y=np.asarray(Image.open(ROOT/r['mask']).convert('L'))>r.get('mask_threshold',0)
        p=predict(model,x)>=.5;row={'id':r['id'],**metrics(p,y)}
        if 'group' in r:row['group']=r['group']
        records.append(row)
        if out:preds[r['id']]=np.packbits(p.reshape(-1));shapes[r['id']]=list(p.shape)
    keys=[k for k,v in records[0].items() if type(v) in (float,int)]
    result={'images':len(records),'mean':{k:float(np.mean([r[k] for r in records])) for k in keys},'per_image':records}
    if out:np.savez_compressed(out/'predictions_packed.npz',**preds);(out/'prediction_shapes.json').write_text(json.dumps(shapes))
    return result
