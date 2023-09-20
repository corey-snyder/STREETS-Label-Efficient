import numpy as np
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from quantitative_evaluation import *

def log_metrics(writer, results_dict, step, validation=False):
    if validation:
        split = 'Validation'
    else:
        split = 'Training'
    writer.add_scalar('{}/Image Metrics/Precision'.format(split), results_dict['Image']['Precision'], step)
    writer.add_scalar('{}/Image Metrics/Recall'.format(split), results_dict['Image']['Recall'], step)
    writer.add_scalar('{}/Image Metrics/F-measure'.format(split), results_dict['Image']['F-measure'], step)
    writer.add_scalar('{}/Total Metrics/Precision'.format(split), results_dict['Total']['Precision'], step)
    writer.add_scalar('{}/Total Metrics/Recall'.format(split), results_dict['Total']['Recall'], step)
    writer.add_scalar('{}/Total Metrics/F-measure'.format(split), results_dict['Total']['F-measure'], step)
    return writer

def unet_train(model, train_dataset, val_dataset, writer, config):
    # set up dataloader
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # set up criteria
    criterion = nn.BCELoss()  
    # set up optimizer
    lr = float(config['lr'])
    n_epochs = config['n_epochs'] 
    val_freq = int(n_epochs/20) # log 20 validation updates
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    milestones = [int(n_epochs*m) for m in config['milestones']]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # identify allowable cropping regions
    if config['crop_size'] != 'None':
        crop_size = int(config['crop_size'])
        max_x, max_y = config['input_shape'][1]-crop_size, config['input_shape'][0]-crop_size
    else:
        crop_size = None
    # training loop
    model = model.to(device)
    for epoch in tqdm(range(n_epochs)):
        train_total = 0
        n_images = 0
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            if crop_size is not None:
                x_coord = np.random.choice(np.arange(max_x))
                y_coord = np.random.choice(np.arange(max_y))
                images = images[:, :, y_coord:y_coord+crop_size, x_coord:x_coord+crop_size]
                targets = targets[:, :, y_coord:y_coord+crop_size, x_coord:x_coord+crop_size]
            images, targets = images.to(device), targets.to(device)
            n_images += images.size(0)
            preds = model(images)
            loss = criterion(preds, targets) 
            train_total += loss.item()
            loss.backward()
            optimizer.step()
        # logging
        writer.add_scalar('Training/Loss', train_total/n_images, epoch+1) 
        if ((epoch+1) % val_freq) == 0:
            model.eval()
            train_results_dict = evaluate_unet(train_dataset, model, device)
            validation_results_dict = evaluate_unet(val_dataset, model, device)
            writer = log_metrics(writer, train_results_dict, epoch+1)
            writer = log_metrics(writer, validation_results_dict, epoch+1, validation=True)
            # prepare for next epoch
            model.train() 
        scheduler.step() 
    return model, writer
