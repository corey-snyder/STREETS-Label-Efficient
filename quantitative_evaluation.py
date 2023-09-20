import numpy as np
import os
import torch

from skimage.io import imread
from skimage.transform import rescale
from skimage.measure import label 
from metrics import *
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
from dataset import *

ANNOTATIONS_GT_PATH = 'Semantic-Annotations'
ANNOTATIONS_IMAGES_PATH = 'Semantic-Images'

def evaluate(predictions, targets):
    thresholds = np.linspace(0, 1, 51, endpoint=True)
    ioa_thresholds = [0.25, 0.5, 0.75, 0.9]
    blob_targets = [get_blobs(t) for t in targets] # separate each GT image into its blob images
    n_blobs = np.sum([len(b) for b in blob_targets])
    # metrics for each threshold
    precisions, recalls, f_measures, blob_recalls, blob_f_measures = [], [], [], [], [] # entire scene metrics
    image_precisions, image_recalls, image_f_measures, image_blob_recalls, image_blob_f_measures = [], [], [], [], [] # image average metrics
    blob_recalls, blob_f_measures = [[] for it in ioa_thresholds], [[] for it in ioa_thresholds] # blob metrics
    image_blob_recalls, image_blob_f_measures = [[] for it in ioa_thresholds], [[] for it in ioa_thresholds]
    for t in thresholds:
        image_precision, image_recall, = 0, 0
        TP, PP, P = 0, 0, 0
        total_correct_blobs, image_blob_recall = [0 for it in ioa_thresholds], [0 for it in ioa_thresholds]
        n_gt = 0
        # compute metrics
        for i in range(len(targets)):
            curr_pred = predictions[i] > t
            curr_TP, curr_PP, curr_P = get_image_level_metrics(curr_pred, targets[i]) 
            if np.sum(targets[i]):
                n_gt += 1
                image_precision += precision(curr_TP, curr_PP)
                image_recall += recall(curr_TP, curr_P)
                # blob metrics
                for j, ioa_threshold in enumerate(ioa_thresholds):
                    n_blobs_correct, blob_recall = get_blob_metrics(curr_pred, blob_targets[i], ioa_threshold)
                    total_correct_blobs[j] += n_blobs_correct
                    image_blob_recall[j] += blob_recall
            TP += curr_TP
            PP += curr_PP
            P += curr_P
        # store results
        image_precisions.append(image_precision/n_gt)
        image_recalls.append(image_recall/n_gt)
        image_f_measures.append(f_measure(image_precision/n_gt, image_recall/n_gt))
        precisions.append(precision(TP, PP))
        recalls.append(recall(TP, P))
        f_measures.append(f_measure(precision(TP, PP), recall(TP, P)))
        # store blob results
        for j in range(len(ioa_thresholds)):
            image_blob_recalls[j].append(image_blob_recall[j]/n_gt)
            image_blob_f_measures[j].append(f_measure(image_precision/n_gt, image_blob_recall[j]/n_gt))
            blob_recalls[j].append(total_correct_blobs[j]/n_blobs)
            blob_f_measures[j].append(f_measure(precision(TP, PP), total_correct_blobs[j]/n_blobs))
    # find best results and return
    idx = np.argmax(f_measures)
    image_idx = np.argmax(image_f_measures)
    results_dict = {'Image': {'Precision': image_precisions[image_idx], 'Recall': image_recalls[image_idx], 'F-measure': image_f_measures[image_idx],
                              'Blob Recall': {str(it): image_blob_recalls[j][image_idx] for j, it in enumerate(ioa_thresholds)},
                              'Blob F-measure': {str(it): image_blob_f_measures[j][image_idx] for j, it in enumerate(ioa_thresholds)}},
                    'Total': {'Precision': precisions[idx], 'Recall': recalls[idx], 'F-measure': f_measures[idx],
                              'Blob Recall': {str(it): blob_recalls[j][idx] for j, it in enumerate(ioa_thresholds)},
                              'Blob F-measure': {str(it): blob_f_measures[j][idx] for j, it in enumerate(ioa_thresholds)}},
                    'Best Total Threshold': thresholds[idx], 'Best Image Threshold': thresholds[image_idx]}
    return results_dict 

def evaluate_unet(dataset, model, device):
    # iterate through dataset to assemble images, GT
    gt_targets = np.array([(dataset[i][1]).squeeze(0).numpy() for i in range(len(dataset))])
    predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image = dataset[i][0].unsqueeze(0).to(device)
            pred = model(image).squeeze(0).squeeze(0).cpu().numpy()
            predictions.append(pred)
    predictions = np.array(predictions)
    # compute metrics
    results_dict = evaluate(predictions, gt_targets)
    return results_dict

def get_blobs(target_image):
        '''
        Extract each connected component or "blob" from target_image. This enables
        per-blob metrics like blob recall.
        '''
        all_blobs = label(target_image, connectivity=1)
        blob_numbers = np.unique(all_blobs)
        if len(blob_numbers) > 1:
            blob_numbers = blob_numbers[blob_numbers>0] # ignore 0
            blobs = np.array([all_blobs==n for n in blob_numbers]) # (N_blobs, H, W)
            return blobs
        else:
            return []
