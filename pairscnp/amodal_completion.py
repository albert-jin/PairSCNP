import torch
from torch.nn import functional as F
from .chroma_consistency import colors,plain

@torch.no_grad()
def amodal_view(x,labels,geometry,generator,step,probability=.75):
    x=plain(x);rgb,mu,sd=colors(x);y=plain(labels).float();g=plain(geometry).float()
    n,_,h,w=rgb.shape;device=rgb.device
    assert y.shape==g.shape==(n,1,h,w) and 0<=probability<=1
    active=(torch.rand(n,1,1,1,device=device,generator=generator)<probability)
    mask=torch.zeros_like(y);curriculum=.5+.5*min(max(step,0)/600.,1.)
    yy=torch.arange(h,device=device)[None,None,:,None]
    xx=torch.arange(w,device=device)[None,None,None,:]
    for slot in range(2):
        if slot==0:
            weights=(g.clamp_min(0)*y).flatten(1)
            weights=torch.where(weights.sum(1,keepdim=True)>0,weights,torch.ones_like(weights))
            centers=torch.multinomial(weights,1,generator=generator).view(n)
            cy,cx=centers//w,centers%w
        else:
            cy=torch.randint(h,(n,),device=device,generator=generator)
            cx=torch.randint(w,(n,),device=device,generator=generator)
        size=torch.randint(36,101,(n,2),device=device,generator=generator).float()*curriculum
        hh=(size[:,0]*h/512).long().clamp(1,h)
        ww=(size[:,1]*w/512).long().clamp(1,w)
        top=torch.minimum((cy-hh//2).clamp_min(0),h-hh)[:,None,None,None]
        left=torch.minimum((cx-ww//2).clamp_min(0),w-ww)[:,None,None,None]
        rect=(yy>=top)&(yy<top+hh[:,None,None,None])&(xx>=left)&(xx<left+ww[:,None,None,None])
        mask=torch.maximum(mask,(rect&active).float())
    # Smooth random appearance, independent of all target/test images.
    texture=F.interpolate(torch.rand(n,3,4,4,device=device,generator=generator),size=(h,w),mode='bilinear',align_corners=False)
    alpha=.75+.25*torch.rand(n,1,1,1,device=device,generator=generator)
    opaque=torch.rand(n,1,1,1,device=device,generator=generator)<.5
    alpha=torch.where(opaque,torch.ones_like(alpha),alpha)*mask
    mixed=(rgb*(1-alpha)+texture*alpha-mu)/sd
    result=torch.where(mask.bool(),mixed,x)
    return result,mask

