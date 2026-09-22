"""Shared DeepLabv3+ backbone and the upstream SCNP segmentation head."""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from . import author_deeplab
from .author_deeplab import semantic_seg
from .author_deeplab.loss import SCNPCEDiceLoss
from .paths import ROOT

# Preserve the upstream implementation while supplying its dispatch alias.
semantic_seg.SCNPCEDice = SCNPCEDiceLoss


class BackboneCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        result = super()._load_model(checkpoint)
        self.audit = {"missing": list(result.missing_keys),
                      "unexpected": list(result.unexpected_keys),
                      "incorrect_shapes": list(result.incorrect_shapes)}
        missing_backbone = [key for key in result.missing_keys
                            if key.startswith("backbone.") and not key.endswith("num_batches_tracked")]
        if missing_backbone or result.incorrect_shapes:
            raise RuntimeError(f"Backbone initialization is incomplete: {self.audit}")
        return result


class PairSCNP(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        cfg = get_cfg()
        author_deeplab.add_deeplab_config(cfg)
        cfg.merge_from_file(str(ROOT / "configs/deeplab_r103.yaml"))
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
        cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "scnpcedice"
        cfg.INPUT.CROP.SIZE = (512, 512)
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.MAX_ITER = 12000
        cfg.SOLVER.WEIGHT_DECAY = 0.1
        cfg.SOLVER.IMS_PER_BATCH = 4
        self.net = build_model(cfg)
        self.load_audit = None
        if weights:
            loader = BackboneCheckpointer(self.net)
            loader.load(str(weights))
            self.load_audit = loader.audit
        self.config_text = cfg.dump()

    @property
    def criterion(self):
        return self.net.sem_seg_head.loss

    def forward(self, x):
        features = self.net.backbone(x)
        logits = self.net.sem_seg_head.layers(features)
        logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
        assert logits.shape[-2:] == x.shape[-2:] == (512, 512)
        return logits
