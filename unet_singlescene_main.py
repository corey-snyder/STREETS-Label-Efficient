from torch.utils.tensorboard import SummaryWriter

import yaml
import json

from argparse import ArgumentParser
from torch.utils.data import Subset
from train import unet_train
from dataset import *
from models import UNet
from yaml import Loader

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to YAML config file')
    args = parser.parse_args()

    # load config file
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
     
    for view in config['views']:
        config['view'] = view 
        for seed in config['seeds']:
            # set random seed
            config['seed'] = seed
            np.random.seed(seed)
            # initialize U-Net
            n_channels, n_classes = 3, 1
            model = UNet(n_channels, n_classes)
            # create datasets
            input_shape, output_shape = config['input_shape'], config['output_shape']
            images_path, gt_path = config['images_path'], config['gt_path']
            dataset = SemanticBFSDataset(images_path, gt_path, view, input_shape, output_shape)
            n_train = config['n_train']
            train_indices = np.random.choice(np.arange(len(dataset)), size=n_train, replace=False)
            val_indices = np.array([i for i in np.arange(len(dataset)) if i not in train_indices])
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices) 
            # set up logging
            tag = 'unet_{}_{}_images_seed{}'.format(view, n_train, seed)
            logging_path = os.path.join(config['log_path'], tag)
            writer = SummaryWriter(logging_path)
            # save config file 
            with open(os.path.join(logging_path, 'config.json'), 'w') as f:
                json.dump(config, f)
            # train model 
            model, writer = unet_train(model, train_dataset, val_dataset, writer, config)
            # save model
            torch.save(model.cpu(), os.path.join(logging_path, 'model.pt'))
