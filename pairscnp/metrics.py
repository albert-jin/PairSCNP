"""Independent test metrics with explicit foreground connectivity conventions."""
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
def betti(mask):
    mask=np.asarray(mask,dtype=bool)
    b0=label(mask,np.ones((3,3),int))[1]
    # Complement of closed pixel squares is four-connected; discard exterior.
    _,n=label(np.pad(~mask,1,constant_values=True),np.array([[0,1,0],[1,1,1],[0,1,0]]))
    return int(b0),int(n-1)
def metrics(pred,target):
    p=np.asarray(pred,dtype=bool);y=np.asarray(target,dtype=bool);inter=np.logical_and(p,y).sum()
    total=p.sum()+y.sum();union=np.logical_or(p,y).sum()
    dice=float(2*inter/total) if total else 1.
    iou=float(inter/union) if union else 1.
    ps=skeletonize(p);ys=skeletonize(y)
    precision=float((ps&y).sum()/max(1,ps.sum()));recall=float((ys&p).sum()/max(1,ys.sum()))
    cldice=2*precision*recall/(precision+recall) if precision+recall else float(not p.any() and not y.any())
    pb=betti(p);yb=betti(y)
    return {'dice':dice,'foreground_iou':iou,'cldice':cldice,'betti0_error':abs(pb[0]-yb[0]),'betti1_error':abs(pb[1]-yb[1]),
            'pred_betti0':pb[0],'true_betti0':yb[0],'pred_betti1':pb[1],'true_betti1':yb[1],
            'author_betti0_error':abs(max(1,pb[0])-max(1,yb[0])),
            'author_betti1_error':abs(pb[1]-yb[1]),
            'author_cldice_defined':bool(ps.any() and ys.any() and precision+recall>0)}
