import torch

def plain(x):return x.as_tensor() if hasattr(x,'as_tensor') else x


def colors(x):
    x=plain(x).float()
    mu=x.new_tensor([.485,.456,.406])[None,:,None,None]
    sd=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x*sd+mu).clamp(0,1),mu,sd


@torch.no_grad()
def paired_view(x,generator):
    rgb,mu,sd=colors(x);n=len(rgb)
    def rand(channels=1):return torch.rand(n,channels,1,1,device=rgb.device,generator=generator)
    # Every draw uses a dedicated stream shared across experimental arms.
    gamma=.65+.85*rand();gain=.6+.8*rand(3)
    z=rgb.clamp_min(1e-6).pow(gamma)*gain
    gray=(z*z.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    saturation=1.5*rand();saturation=torch.where(rand()<.35,torch.zeros_like(saturation),saturation)
    z=gray+saturation*(z-gray)
    permutation=torch.rand(n,3,device=z.device,generator=generator).argsort(1)
    permuted=z.gather(1,permutation[:,:,None,None].expand_as(z))
    z=torch.where(rand()<.25,permuted,z)
    return (z.clamp(0,1)-mu)/sd


def structure_js(z1,z2,labels,geometry):
    """Symmetric JS, normalized within each present class and image.

    GT geometry weights lie in [1,5]. Empty classes contribute no denominator
    or loss. No cross-image normalization, pseudo labels or test data are used.
    """
    p=plain(z1).float().softmax(1).clamp_min(1e-7)
    q=plain(z2).float().softmax(1).clamp_min(1e-7)
    m=.5*(p+q)
    js=(.5*(p*(p.log()-m.log())+q*(q.log()-m.log()))).sum(1,keepdim=True).clamp_min(0)
    y=plain(labels).float();w=(1+4*plain(geometry).float().clamp(0,1)).detach()
    values=[];present=[]
    for mask in (y,1-y):
        wm=w*mask;denom=wm.sum((1,2,3));ok=denom>0
        values.append((js*wm).sum((1,2,3))/denom.clamp_min(1e-12))
        present.append(ok.float())
    return ((values[0]+values[1])/(present[0]+present[1]).clamp_min(1)).mean()

