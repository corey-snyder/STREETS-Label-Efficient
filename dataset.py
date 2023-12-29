import numpy as np
import torch
import os
import json

from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave 

class SemanticBFSDataset(Dataset):
    def __init__(self, images_path, gt_path, view, input_shape, output_shape):
        self.images_path = images_path
        self.top_trim = 15*int(input_shape[0]/240) # top trim should be 15 if images are downsampled by 2
        self.gt_indices = None # not sequence data
        self.gt_path = gt_path
        self.view = view
        self.input_shape, self.output_shape = input_shape, output_shape 
        # load label time dictionary
        with open(os.path.join('labeling-times', 'Semantic-Annotation-Times.json'), 'r') as f:
            label_time_dict = json.load(f)
        # gather image file paths
        initial_file_paths = []
        view_path = os.path.join(gt_path, view)
        if os.path.exists(view_path):
            for date in os.listdir(view_path):
                date_path = os.path.join(view_path, date)
                initial_file_paths += [os.path.join(view, date, n) for n in os.listdir(date_path)]
        # load images and targets if original image is available
        self.file_paths = []
        self.images, self.targets, self.label_times = [], [], []
        for f in initial_file_paths:
            full_image_path = os.path.join(self.images_path, f.replace('.png', '.jpg'))
            # input image
            if os.path.exists(full_image_path):
                self.file_paths.append(f)
                input_image = resize(imread(full_image_path), self.input_shape) 
                input_image[:self.top_trim] = 0
                self.images.append(torch.from_numpy(input_image).permute(2, 0, 1).float()) # make image a Tensor in (C, H, W) format
                # target image
                target_image = resize(imread(os.path.join(self.gt_path, f)), self.output_shape)
                target_image[:self.top_trim] = 0
                target_image = target_image > 0
                self.targets.append(torch.from_numpy(target_image).unsqueeze(0).float()) # make image a Tensor, shape (1, H, W)
                # label time
                name_elements = full_image_path.split(os.sep)
                self.label_times.append(float(label_time_dict[view][name_elements[-1].replace('.jpg', '.png')])) 
        self.file_paths = np.array(self.file_paths) 
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx): 
        return self.images[idx], self.targets[idx]

class BBoxBFSDataset(Dataset):
    def __init__(self, images_path, gt_path, view, input_shape, output_shape): 
        self.images_path = images_path
        self.top_trim = 15*int(input_shape[0]/240) # top trim should be 15 if images are downsampled by 2
        self.gt_indices = None # not sequence data
        self.gt_path = gt_path
        self.view = view
        self.input_shape, self.output_shape = input_shape, output_shape 
        # load bbox GT dictionary
        with open(os.path.join(gt_path, 'bbox-annotations.json'), 'r') as f:
            all_gt = json.load(f)
        # load label time dictionary
        with open(os.path.join('labeling-times', 'BBox-Annotation-Times.json'), 'r') as f:
            label_time_dict = json.load(f) 

        # load images and targets
        self.file_paths = []
        self.images, self.targets, self.label_times = [], [], [] 
            
        view_dict = all_gt[view]
        for date in view_dict:
            date_dict = view_dict[date]
            for image_name in date_dict:        
                full_image_path = os.path.join(self.images_path, view, date, image_name)
                if os.path.exists(full_image_path):
                    self.file_paths.append(f)
                    # input image
                    input_image = imread(full_image_path)
                    original_H, original_W = input_image.shape[0], input_image.shape[1] 
                    input_image = resize(input_image, self.input_shape)
                    input_image[:self.top_trim] = 0
                    self.images.append(torch.from_numpy(input_image).permute(2, 0, 1).float())
                    # ground-truth
                    original_bboxes = all_gt[view][date][image_name]
                    height_dilation = self.output_shape[0]/original_H
                    width_dilation = self.output_shape[1]/original_W
                    # adjust bbox coordinates (in [xmin, ymin, xmax, ymax] format)
                    save_bboxes = [[b[0]*width_dilation, b[1]*height_dilation, b[2]*width_dilation, b[3]*height_dilation] for b in original_bboxes]
                    # self.targets[i] is a list of bounding-box coordinates
                    # self.targets[i][j] gives (xmin, ymin, xmax, ymax) for box j from image i
                    self.targets.append(save_bboxes)
                    # get labeling time
                    self.label_times.append(float(label_time_dict[view][image_name]))
        self.file_paths = np.array(self.file_paths)
        self.label_times = np.array(self.label_times) 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx): 
        file_path = self.file_paths[idx]
        return self.images[idx], self.targets[idx]

class MultiSceneDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets # list of datasets to concatenate together
        self.dataset_lengths = np.array([len(d) for d in datasets]) # (N_datasets,)
        self.dataset_limits = np.array([np.sum(self.dataset_lengths[:i+1]) for i in range(len(datasets))]) # (N_datasets,)
        # set labeling times
        if datasets[0].label_times is not None:
            self.label_times = np.array([t for d in datasets for t in d.label_times])
        else:
            self.label_times = None
        # set gt indices:
        if datasets[0].gt_indices is not None:
            self.gt_indices = np.array([gi for d in datasets for gi in d.gt_indices])
        else:
            self.gt_indices = None
        self.top_trim = self.datasets[0].top_trim

    def __len__(self):
        return np.sum(self.dataset_lengths)

    def __getitem__(self, idx):
        # identify dataset to get item from
        less_than_limit = idx < self.dataset_limits
        dataset_idx = np.flatnonzero(less_than_limit)[0] # choose first dataset that extends concatenated dataset beyond idx
        # identify index within selected dataset to get item
        dataset_relative_idx = idx - np.sum(self.dataset_lengths[:dataset_idx])
        return self.datasets[dataset_idx][dataset_relative_idx]
 
