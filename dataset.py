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
                    self.file_paths.append(full_image_path)
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
                    # trim bounding boxes within top_trim range or remove, if necessary
                    final_bboxes = []
                    for b in save_bboxes:
                        xmin, ymin, xmax, ymax = b
                        # outside of top trim
                        if ymax > self.top_trim:
                            final_bboxes.append([xmin, max(ymin, self.top_trim), xmax, ymax])
                    # self.targets[i] is a list of bounding-box coordinates
                    # self.targets[i][j] gives (xmin, ymin, xmax, ymax) for box j from image i
                    self.targets.append(final_bboxes)
                    # get labeling time
                    self.label_times.append(float(label_time_dict[view][image_name]))
        self.file_paths = np.array(self.file_paths)
        self.label_times = np.array(self.label_times) 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx): 
        file_path = self.file_paths[idx]
        return self.images[idx], self.targets[idx]

class BBoxSequenceDataset(Dataset):
    def __init__(self, images_path, gt_path, view, input_shape, output_shape, stride):
        self.images_path = images_path
        self.stride = stride 
        self.top_trim = 15*int(input_shape[0]/240) 
        self.gt_path = gt_path
        self.view = view
        self.input_shape, self.output_shape = input_shape, output_shape
        # load bbox GT dictionary
        with open(os.path.join(gt_path, 'bbox-annotations.json'), 'r') as f:
            all_gt = json.load(f)
        # load label time dictionary
        with open(os.path.join('labeling-times', 'BBox-Annotation-Times.json'), 'r') as f:
            label_time_dict = json.load(f) 
    
        file_paths = [] # general paths that can be pre-pended with input or rpca directories
        gt_indices = [] # index of image with GT within each image block for evaluation
        # maintain separate list for each GT image to grab block of (stride) images
        image_data_path = os.path.join(images_path, view)
        for date in os.listdir(image_data_path):
            date_path = os.path.join(image_data_path, date)
            date_file_paths = []
            curr_gt_date_path = os.path.join(gt_path, view, date)
            # check if there are ground-truth images for this view-date combination
            if date in all_gt[view]:
                for name in all_gt[view][date]:
                    # find relative index within day's image data to retrieve image sequence with GT image
                    curr_image_date_path = os.path.join(images_path, view, date)
                    is_match = [True if n==name else False for n in sorted(os.listdir(curr_image_date_path))]
                    # extract file names if there is a match and at least (stride) images on that day
                    if np.sum(is_match) and len(is_match) >= self.stride: # if np.sum(is_match):
                        relative_gt_idx = np.flatnonzero(is_match)[0]
                        if relative_gt_idx >= self.stride:
                            curr_file_names = sorted(os.listdir(curr_image_date_path))[relative_gt_idx:relative_gt_idx-self.stride:-1]
                            gt_idx = 0
                        else:
                            curr_file_names = sorted(os.listdir(curr_image_date_path))[self.stride-1::-1]
                            gt_idx = self.stride-relative_gt_idx-1
                        file_paths.append([os.path.join(view, date, n) for n in curr_file_names]) # names with .jpg extension
                        gt_indices.append(gt_idx) 

        # load image data
        self.file_paths = file_paths # list of lists (N_labeled) lists of length (stride)
        self.gt_indices = gt_indices
        self.images = [] # grayscale images
        self.targets = [] # GT targets
        self.label_times = []
        for i, curr_list in enumerate(self.file_paths): 
            self.images.append(torch.stack([torch.from_numpy(resize(imread(os.path.join(self.images_path, f)),
                                                                        self.output_shape)).permute(2, 0, 1) for f in curr_list], dim=0))
            self.images[i][:, :, :self.top_trim] = 0
            gt_idx = self.gt_indices[i]
            # ground-truth
            file_path = curr_list[gt_idx]
            name_elements = file_path.split(os.sep)
            date = name_elements[1]
            image_name = name_elements[2]
            original_H, original_W = self.images[i][0].size(1), self.images[i][0].size(2)
            original_bboxes = all_gt[self.view][date][image_name]
            height_dilation = self.output_shape[0]/original_H
            width_dilation = self.output_shape[1]/original_W
            # adjust bbox coordinates (in [xmin, ymin, xmax, ymax] format)
            save_bboxes = [[b[0]*width_dilation, b[1]*height_dilation, b[2]*width_dilation, b[3]*height_dilation] for b in original_bboxes]
            # trim bounding boxes within top_trim range or remove, if necessary
            final_bboxes = []
            for b in save_bboxes:
                xmin, ymin, xmax, ymax = b
                # outside of top trim
                if ymax > self.top_trim:
                    final_bboxes.append([xmin, max(ymin, self.top_trim), xmax, ymax])
            # self.targets[i] is a list of bounding-box coordinates
            # self.targets[i][j] gives (xmin, ymin, xmax, ymax) for box j from image i
            self.targets.append(final_bboxes)
            # label time
            self.label_times.append(float(label_time_dict[view][image_name]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_batch = self.images[idx].float()
        target = self.targets[idx] 
        gt_index = self.gt_indices[idx]
        return (image_batch, target, gt_index)


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

if __name__ == '__main__':
    images_path = 'BBox-Images'
    gt_path = 'BBox-Annotations'
    view = 'Almond at Washington East'
    input_shape = [480, 720]
    output_shape = input_shape
    stride = 6
    full_images_path = '../STREETS-unsupervised-bfs/ImageData'
    dataset = BBoxBFSDataset(images_path, gt_path, view, input_shape, output_shape)
    sequence_dataset = BBoxSequenceDataset(full_images_path, gt_path, view, input_shape, output_shape, stride)
    images, target, gt_index = sequence_dataset[0]
    print(images.shape, target, gt_index)
