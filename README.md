# STREETS-Label-Efficient
This repository provides helpful starter code to run baseline single-scene background-foreground separation (BFS) experiments on the STREETS dataset with a focus on working with a limited annotation budget.

## Downloading the data
After cloning this repository, you should download the ``Semantic-Annotations.zip`` and ``Semantic-Images.zip`` file from our shared Box folder. If you plan to work locally on your own machine, you can unzip these folders to the same path as this repository. If you would like to use Google Colab, you should unzip the files locally and then upload all the images to a dedicated folder on your Google Drive to prevent image loss from unzipping within Google Drive.

## Overview of provided codes and files
A brief explanation of each provided file is given below.

### ``dataset.py``
### ``models.py``
### ``train.py``
### ``quantitative_evaluation.py``
### ``singlescene_benchmark.py``
### ``unet_singlescene_main.py``
### ``config.yaml``

## Running an experiment
The intended workflow for running baseline experiments should be fairly straightforward where very little needs to be edited within the files to run a simple baseline experiment. You may write your own codes or make additional tweaks as you explore your own experiments.
### Setting ``config.yaml``
The ``config.yaml`` file contains the key information that is used for defining each experiment. You will edit the information in here to provide the "settings" of your experiment. The first category of settings is the "learning parameters" for training the U-Net model. You may find you want to adjust these in the future, but the provided settings are a good starting point and you likely will not need to edit them immediately.

The remaining parameters are up to how you want to run your experiment:
* ``images_path``: Path to where the annotated RGB images are stored. You will need to change this to an appropriate mounted Google Drive path where you uploaded the image data if you work on Google Colab.
* ``gt_path``: Path to where the pixel-level annotations are stored corresponding to the images in the ``images_path`` folder. Like ``images_path``, you will need to change this if you work on Google Colab
* ``seeds``: You can specify a list of random seed values to run multiple trials of each experiment. This is good practice since there will be some statistical variance between different choices of random seed, e.g. from which labeled images are used in each trial. I suggest picking simple seed numbers, e.g. 1, 2, 3, instead of oddly specific numbers, e.g. 61732, since specific numbers make it look like results are being herded and made to look nicer than they should be.
* ``views``: This is a list of the names of the scenes or views you want to train models for. You may specify one or more scene names. You can find the names of scenes by looking at the subfolders within the ``Semantic-Images`` or ``Semantic-Annotations`` folders.
* ``n_train``: This is the number of training images that will be used to train the U-Net model for each scene. Keep this number below 30 since we only have 50 annotated images per scene and need to reserve some number of images for evaluation.
* ``input_shape``: This is the size (height by width, H x W) that the input images will be loaded to. Most available images are 480x720, but you may consider using a smaller size like 240x360 to speed up your experiments.
* ``output_shape``: This is the size (H x W) that the ground-truth images will be loaded to. This size can be different from ``input_shape``, but I would recommend keeping the two the same.
* ``crop_size``: Optional argument to specify a square random cropping size to enable image augmentation while training. You may leave this as ``None`` to have no cropping or pick a number less than the image height. The smaller the crop size, the longer you will likely need to train your models. I recommend not cropping images to start, but you may explore it in the future.
* ``log_path``: Path to where logs will be saved. These logs include the model weights that can be loaded later for evaluation purposes and Tensorboard logs. More on Tensorboard later! 
### Training models
### Evaluating models
### Processing and examining results
