# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import random
from PIL import ImageFilter

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, 
                 transforms, strong_transforms=None,
                 return_masks=False, cache_mode=False, 
                 local_rank=0, local_size=1
                 ):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self._transforms_strong = strong_transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        img_strong_aug = None
        if self._transforms_strong is not None:
            img_strong_aug = self._transforms_strong(img)
        if self._transforms is not None:
            img, img_strong_aug, target = self._transforms(img, img_strong_aug, target)
        return img, img_strong_aug, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def compute_multi_scale_scales(resolution, patch_size, expanded_scales=False):

    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * 4)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [int(scale * patch_size * 4) for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * 4]  # ensure minimum image size
    return proposed_scales

def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800] if not is_vit_backbone \
    #      else compute_multi_scale_scales(644, patch_size, expanded_scales=True)
    # scales = [448, 476, 504, 532, 560, 616, 672, 700, 728, 756, 784]

    if image_set == 'train':

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([args.img_size], max_size=1333),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize([644], max_size=1333),
            #     ])
            # ),
            normalize,
        ])

    if image_set == 'val':
        
    
        return T.Compose([
            T.RandomResize([args.img_size], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def make_coco_strong_transforms(image_set):
    augmentation = []

    if image_set == 'train':
        augmentation.append(
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(T.RandomGrayscale(p=0.2))
        augmentation.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = T.Compose(
            [
                T.ToTensor(),
                # T.RandomErasing(
                #     p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                # ),
                # T.RandomErasing(
                #     p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                # ),
                # T.RandomErasing(
                #     p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                # ),
                T.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

        return T.Compose(augmentation)
    
    if image_set == 'val':
        return None

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": ("/input0/xView/images", "/input0/xView/annotations/train_3c.json"),
        # "train": ("/input0/DOTA/train/images" , "/input0/DOTA/annotations/train_3c.json"),
        "val": ("/input0/DOTA/val/images" , "/input0/DOTA/annotations/val_3c.json"),
        # "val": ("/input0/DOTA/val/images" , "/input0/DOTA/annotations/val_3c.json"),
        # "train": ("/input1/split640_NWPU_VHR_10/train/images" , "/input1/split640_NWPU_VHR_10/annotations/instances_train.json"),
        # "train": ("/input1/split640_DIOR_subset/train/images", "/input1/split640_DIOR_subset/annotations/instances_train.json"),
        # "val": ("/input1/split640_DIOR_subset/val/images", "/input1/split640_DIOR_subset/annotations/instances_val.json"),
        # "val": ("/input1/split640_HRRSD_subset/val/images", "/input1/split640_HRRSD_subset/annotations/instances_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
