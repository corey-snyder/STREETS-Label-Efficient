import os
import torch
import json
import numpy as np

from quantitative_evaluation import *
from dataset import *
from torch.utils.data import Subset
from pprint import pprint
from tqdm import tqdm


def test_unet(folder_path, views, n_train_list):
    final_results_dict = {n: {v: {w: [] for w in views} for v in views} for n in n_train_list}
    for exp in tqdm(os.listdir(folder_path)):
        experiment_path = os.path.join(folder_path, exp)
        # load model
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = torch.load(os.path.join(experiment_path, 'model.pt'))
        model = model.to(device)
        model.eval()
        # load config
        with open(os.path.join(experiment_path, 'config.json'), 'r') as f:
            config = json.load(f)
        # identify training view, seed, number of training images
        training_view = config['view']
        seed = config['seed']
        np.random.seed(seed)
        n_train = int(config['n_train'])
        # initialize training/validation dataset
        images_path, gt_path, input_shape, output_shape = config['images_path'], config['gt_path'], config['input_shape'], config['output_shape']
        dataset = SemanticBFSDataset(images_path, gt_path, training_view, input_shape, output_shape)
        training_indices = np.random.choice(np.arange(len(dataset)), size=n_train, replace=False)
        validation_indices = np.array([i for i in np.arange(len(dataset)) if i not in training_indices])
        val_dataset = Subset(dataset, validation_indices)
        # evaluate training scene
        validation_results_dict = evaluate_unet(val_dataset, model, device)
        final_results_dict[n_train][training_view][training_view].append(validation_results_dict)
        # evaluate test scenes
        for test_view in views:
            if test_view != training_view:
                test_dataset = SemanticBFSDataset(images_path, gt_path, test_view, input_shape, output_shape)
                test_results_dict = evaluate_unet(test_dataset, model, device)
                final_results_dict[n_train][training_view][test_view].append(test_results_dict) 
    # return results 
    return final_results_dict


if __name__ == '__main__':
    save_path = 'single-scene-benchmark'
    folders = ['MyLogs']
    views = ['Aptakisic at Leider West', 'Deerfield at Saunders South', 'Dilleys at Stearns School South', 'IL 21 at IL 60 West', 'US 45 at Deerpath North']
    n_train_list = [5, 10, 20]
    for folder in folders:
        folder_path = os.path.join('logs', folder)
        print('Compiling results from: {}...'.format(folder))
        results_dict = test_unet(folder_path, views,n_train_list) 
        # save results
        with open(os.path.join(save_path, '{}.json'.format(folder)), 'w') as f:
            json.dump(results_dict, f)
