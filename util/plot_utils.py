# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


# --- BEGIN NEW FUNCTION ---
def visualize_bboxes(pil_img, results, ground_truth, output_path, class_names, score_thresh=0.7):
    """
    Visualizes ground truth and predicted bounding boxes on an image and saves it.

    Args:
        pil_img (PIL.Image.Image): The input image.
        results (dict): A dictionary with 'scores', 'labels', 'boxes' for predictions.
        ground_truth (dict): A dictionary with 'labels', 'boxes' for ground truth.
        output_path (str): Path to save the visualized image.
        class_names (list): A list of class names, indexed by class ID.
        score_thresh (float): Confidence threshold to filter predictions.
    """
    draw = PIL.ImageDraw.Draw(pil_img)
    
    try:
        font = PIL.ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = PIL.ImageFont.load_default()

    # Draw ground truth boxes in green
    if 'boxes' in ground_truth and 'labels' in ground_truth:
        for box, label_id in zip(ground_truth['boxes'], ground_truth['labels']):
            draw.rectangle(box.tolist(), outline='green', width=3)
            label_text = f"GT: {class_names[label_id.item()]}"
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            draw.rectangle(text_bbox, fill="green")
            draw.text((box[0], box[1]), label_text, fill='white', font=font)

    # Draw predicted boxes in red
    if 'scores' in results and 'labels' in results and 'boxes' in results:
        for score, label_id, box in zip(results['scores'], results['labels'], results['boxes']):
            if score > score_thresh:
                draw.rectangle(box.tolist(), outline='red', width=3)
                label_text = f"Pred: {class_names[label_id.item()]}: {score:.2f}"
                text_bbox = draw.textbbox((box[0], box[1] - 15), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((box[0], box[1] - 15), label_text, fill='white', font=font)
                
    pil_img.save(output_path)
# --- END NEW FUNCTION ---

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs



