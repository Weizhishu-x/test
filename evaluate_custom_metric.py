import json
import numpy as np
import argparse
from collections import defaultdict

def calculate_area(box):
    """计算单个box的面积 [x, y, w, h]"""
    if box[2] < 0 or box[3] < 0:
        return 0.0
    return box[2] * box[3]

def calculate_intersection(box_a, box_b):
    """计算两个box的交集面积 [x, y, w, h]"""
    box_a_xyxy = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
    box_b_xyxy = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]
    ix1 = max(box_a_xyxy[0], box_b_xyxy[0])
    iy1 = max(box_a_xyxy[1], box_b_xyxy[1])
    ix2 = min(box_a_xyxy[2], box_b_xyxy[2])
    iy2 = min(box_a_xyxy[3], box_b_xyxy[3])
    iw = max(ix2 - ix1, 0.)
    ih = max(iy2 - iy1, 0.)
    return iw * ih

def calculate_ap(recalls, precisions):
    """计算AP（平均精度），使用面积插值法"""
    recalls = np.concatenate(([0.], recalls))
    precisions = np.concatenate(([1.], precisions))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap

def evaluate_predictions_for_class(ground_truths, predictions, metric='iou', iou_threshold=0.5):
    """
    为单个类别和单个IoU阈值评估预测结果。
    此函数严格模拟pycocotools的匹配逻辑。
    """
    
    # 按图像ID对真值和预测进行分组
    gt_by_img = defaultdict(list)
    for img_id, boxes in ground_truths.items():
        for i, box in enumerate(boxes):
            gt_by_img[img_id].append({'bbox': box, 'detected': False})

    preds_by_img = defaultdict(list)
    for pred in predictions:
        preds_by_img[pred['image_id']].append(pred)

    tp = []
    fp = []
    total_gt_count = sum(len(v) for v in ground_truths.values())

    # 按置信度从高到低排序所有预测
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)

    for pred in sorted_preds:
        img_id = pred['image_id']
        pred_box = pred['bbox']
        
        gt_list = gt_by_img.get(img_id, [])
        if not gt_list:
            fp.append(1)
            tp.append(0)
            continue

        gt_boxes = np.array([g['bbox'] for g in gt_list])
        
        # 计算与所有真值框的度量
        intersections = np.array([calculate_intersection(pred_box, gt_box) for gt_box in gt_boxes])
        pred_area = calculate_area(pred_box)

        if metric == 'iou':
            gt_areas = np.array([calculate_area(gt_box) for gt_box in gt_boxes])
            unions = pred_area + gt_areas - intersections
            metrics = intersections / (unions + 1e-8)
        elif metric == 'iop':
            metrics = intersections / (pred_area + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best_gt_idx = -1
        max_metric = iou_threshold # 必须大于等于阈值

        # 寻找最佳匹配
        for i, gt in enumerate(gt_list):
            if metrics[i] > max_metric:
                max_metric = metrics[i]
                best_gt_idx = i
        
        if best_gt_idx != -1:
            if not gt_list[best_gt_idx]['detected']:
                gt_list[best_gt_idx]['detected'] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)

    # 计算AP
    if total_gt_count == 0:
        return 0.0 if len(predictions) > 0 else 1.0

    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(fp)
    recalls = acc_tp / total_gt_count
    precisions = acc_tp / (acc_fp + acc_tp + 1e-8)
    
    return calculate_ap(recalls, precisions)


def main(args):
    print("加载真值文件...")
    with open(args.gt_file, 'r') as f:
        gt_data = json.load(f)

    print("加载预测文件...")
    with open(args.preds_file, 'r') as f:
        preds_data = json.load(f)

    print("整理数据...")
    ground_truths_by_cat = defaultdict(lambda: defaultdict(list))
    for ann in gt_data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        cat_id = ann['category_id']
        img_id = ann['image_id']
        ground_truths_by_cat[cat_id][img_id].append(ann['bbox'])
    
    predictions_by_cat = defaultdict(list)
    for pred in preds_data:
        cat_id = pred['category_id']
        predictions_by_cat[cat_id].append(pred)
    
    category_ids = sorted(ground_truths_by_cat.keys())
    
    print(f"\n开始为 {len(category_ids)} 个类别计算 AP@0.5 ...")
    
    all_aps_iou = []
    all_aps_iop = []

    print("-" * 55)
    print(f"{'类别 ID':<10} | {'AP@IoU>0.5':<15} | {'AP@IoP>0.5':<15} | {'差异':<10}")
    print("-" * 55)

    for cat_id in category_ids:
        gt_for_cat = ground_truths_by_cat.get(cat_id, {})
        preds_for_cat = predictions_by_cat.get(cat_id, [])
        
        ap_iou = evaluate_predictions_for_class(gt_for_cat, preds_for_cat, metric='iou', iou_threshold=0.5)
        all_aps_iou.append(ap_iou)
        
        ap_iop = evaluate_predictions_for_class(gt_for_cat, preds_for_cat, metric='iop', iou_threshold=0.5)
        all_aps_iop.append(ap_iop)

        diff = ap_iop - ap_iou
        print(f"{cat_id:<10} | {ap_iou:<15.4f} | {ap_iop:<15.4f} | {diff:^+10.4f}")

    mean_ap_iou = np.mean(all_aps_iou) if all_aps_iou else 0
    mean_ap_iop = np.mean(all_aps_iop) if all_aps_iop else 0
    total_diff = mean_ap_iop - mean_ap_iou

    print("-" * 55)
    print("\n" + "=" * 20 + " 总结 " + "=" * 20)
    print(f"标准 mAP @ IoU > 0.5 = {mean_ap_iou:.4f}")
    print(f"诊断 mAP @ IoP > 0.5 = {mean_ap_iop:.4f}")
    print(f"总体差异: {total_diff:+.4f}")
    print("=" * 45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate and Compare mAP using IoU and IoP metrics.")
    parser.add_argument('--preds_file', default="/openbayes/home/wzs/Deformable-DETR-main/predictions.json", type=str)
    parser.add_argument('--gt_file', default="/input0/DOTA/annotations/val_3c.json", type=str)
    args = parser.parse_args()
    main(args)