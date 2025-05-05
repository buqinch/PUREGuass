from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
# from .backbone_dinov2 import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_croco_mamba import AsymmetricVitMambaCroCo
# from .backbone_croco_mamba import AsymmetricMambaCroCo, BackboneCrocoCfg
# from .backbone_croco_DIFF import AsymmetricCroCoDIFF, BackboneCrocoCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "vit_mamba": AsymmetricVitMambaCroCo,
    # "croco_DIFF": AsymmetricCroCoDIFF,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
