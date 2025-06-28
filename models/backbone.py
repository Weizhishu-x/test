# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .projector import MultiScaleProjector

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

from transformers import AutoBackbone
from peft import get_peft_model, LoraConfig

class DINOv2Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        dinov2_model_name = args.backbone 
        
        self.feature_extraction_layers = [2, 5, 8, 11]
        # self.feature_extraction_layers = [9, 19, 29, 39]

        # self.projector_scale = [2.0, 1.0, 0.5, 0.25]
        self.projector_scale = [4.0, 2.0, 1.0, 0.5]
        self.dinov2 = AutoBackbone.from_pretrained(
                dinov2_model_name,
                out_features=[f"stage{i}" for i in self.feature_extraction_layers],
                output_attentions=True,
                return_dict=True  # 确保输出是字典形式，方便访问
            )
        
        config = self.dinov2.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        # 创建LoraConfig
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            use_dora=True,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        
        # 应用PEFT
        self.dinov2 = get_peft_model(self.dinov2, peft_config)
        
        print("PEFT model created. Trainable parameters:")
        self.dinov2.print_trainable_parameters()



        self.strides = [8, 16, 32, 64]
        self.num_channels = [self.hidden_size] * len(self.feature_extraction_layers)
        


        self.projector = MultiScaleProjector(
            in_channels=self.num_channels,
            out_channels=256,
            scale_factors=self.projector_scale,
            layer_norm=False,
            rms_norm=False,
        )

    
    def forward(self, tensor_list: NestedTensor):
        
        x = tensor_list.tensors
        
        outputs = self.dinov2(x)
        feats = list(outputs.feature_maps)
        feats = self.projector(feats) if hasattr(self, 'projector') else feats
        attention_maps = [outputs.attentions[i] for i in self.feature_extraction_layers]
        out: Dict[str, NestedTensor] = {}
        for i, feat in enumerate(feats):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[f"feature_{i}"] = NestedTensor(feat, mask)
        return out, attention_maps


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs, attention_maps = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, attention_maps, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if 'dinov2' in args.backbone:
        backbone = DINOv2Backbone(args)
    else:     
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
