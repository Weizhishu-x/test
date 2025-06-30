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
import copy
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .projector import MultiScaleProjector
from transformers import AutoBackbone
from peft import get_peft_model, LoraConfig

class DINOv2Backbone(nn.Module):
    def __init__(self, args, peft=False):
        super().__init__()
        dinov2_model_name = args.backbone 
        
        # self.feature_extraction_layers = [2, 5, 8, 11]
        # self.feature_extraction_layers = [9, 19, 29, 39]
        # self.projector_scale = [2.0, 1.0, 0.5, 0.25]
        
        self.feature_extraction_layers = args.feature_extraction_layers
        self.projector_scale = args.projector_scale
        self.dinov2 = AutoBackbone.from_pretrained(
                dinov2_model_name,
                out_features=[f"stage{i}" for i in self.feature_extraction_layers],
                output_attentions=False, 
                return_dict=True  
            )
        
        config = self.dinov2.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        if is_main_process() and peft:
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
        else:
            for param in self.dinov2.parameters():
                param.requires_grad = False


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

        out: List[NestedTensor] = []
        for i, feat in enumerate(feats):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out
    
    def forward_freezed(self, tensor_list: NestedTensor):
        """
        专门为MAE任务设计的前向传播方法。
        它处理一个批次的完整图像，并返回编码器的输出特征。

        Args:
            tensor_list (NestedTensor): 输入的完整图片张量, 包含图像和掩码。

        Returns:
            out (Dict[str, NestedTensor]): 编码器对可见块处理后的输出特征。
        """
        x = tensor_list.tensors
        B, _, _, _ = x.shape
        outputs = self.dinov2(x[B // 2:])  # 只处理目标域图像
        feats = [outputs.feature_maps[-1]]
        feats = self.projector(feats * len(self.feature_extraction_layers))

        out: List[NestedTensor] = []
        for i, feat in enumerate(feats):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return feats

    def forward_mae(self, images, mask_ratio=0.75):
        """
        专门为MAE任务设计的前向传播方法。
        它处理一个批次的完整图像，在内部进行掩码，并只通过编码器处理可见部分。

        Args:
            images (torch.Tensor): 输入的完整图片张量, 形状 (B, C, H, W)。
            mask_ratio (float): 掩码比例。

        Returns:
            tuple:
                - latent (torch.Tensor): 编码器对可见块处理后的输出特征。
                - mask (torch.Tensor): 用于重建的二进制掩码。
                - ids_restore (torch.Tensor): 用于恢复块顺序的索引。
        """

        B, _, H, W = images.shape
        device = images.device
        images = images[B // 2:]  # 只处理目标域图像
        
        # 1. 获取嵌入层输出 
        # 调用完整的 dinov2 forward，但只为了获取嵌入层的输出
        # 这样可以确保 PEFT/LoRA 的逻辑被正确执行
        outputs = self.dinov2(images, output_hidden_states=True, return_dict=True)
        all_tokens_embedded = outputs.hidden_states[0]  # 第0个hidden_state就是嵌入层输出 (B, N+1, D)

        # 2. 分离 [CLS] token 和图像块
        patch_tokens = all_tokens_embedded[:, 1:, :]  # (B, N, D)
        num_patches = patch_tokens.shape[1]

        # 3. 生成掩码 
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(B // 2, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # 4. 只选择可见块
        visible_patches_embedded = torch.gather(
            patch_tokens, 
            dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )

        # 5. 将可见块送入编码器
        # 假设 self.dinov2.encoder 可以被直接调用
        # 如果 self.dinov2 是 PeftModel, 可能需要 self.dinov2.base_model.encoder
        encoder_outputs = self.dinov2.encoder(visible_patches_embedded, return_dict=True)
        latent = encoder_outputs.last_hidden_state  # (B, len_keep, D)

        # 6. 生成二进制掩码 (供解码器使用)
        mask = torch.ones(B // 2, num_patches, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return latent, mask, ids_restore


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs:
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if 'dinov2' in args.backbone:
        backbone = DINOv2Backbone(args, peft=True)
    else:     
        raise NotImplementedError(f"Backbone {args.backbone} is not implemented.")
    model = Joiner(backbone, position_embedding)
    backbone_freezed = DINOv2Backbone(args, peft=False)
    backbone_freezed.projector = copy.deepcopy(model[0].projector)
    for param in backbone_freezed.parameters():
        param.requires_grad = False
    
    return model, backbone_freezed
