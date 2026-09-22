"""Fixed foreground/background structural prior, computed before cropping."""
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize

def structure_prior(mask):
    y = np.asarray(mask, dtype=bool)
    geometry = np.zeros(y.shape, np.float32)
    for region in (y, ~y):
        local = skeletonize(region).astype(np.float32) / (1 + distance_transform_edt(region))
        geometry += maximum_filter(local, size=3) * region
    geometry[[0, -1], :] = 0
    geometry[:, [0, -1]] = 0
    return geometry
