# STREETS-Label-Efficient
This repository provides helpful starter code to run baseline single-scene background-foreground separation (BFS) experiments on the STREETS dataset with a focus on working with a limited annotation budget.

## Downloading the data
After cloning this repository, you should download the ``Semantic-Annotations.zip`` and ``Semantic-Images.zip`` file from our shared Box folder. If you plan to work locally on your own machine, you can unzip these folders to the same path as this repository. If you would like to use Google Colab, you should unzip the files locally and then upload all the images to a dedicated folder on your Google Drive to prevent image loss from unzipping within Google Drive.

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
Once the ``config.yaml`` file is set to your liking, training models is very simple. You only need to run the following command in your terminal or in Google Colab:

``python3 unet_singlescene_main.py --config_path config.yaml``

This will execute all the training runs and save the experiment models and logs to the ``log_path`` folder for each pair of (random seed, view). When picking your number of random seeds and views/scenes. Consider that you will be training "number of seeds"x"number of scenes models". On GPU, each (seed, view) trial should take only 5-10 minutes so you can base your config files off this information.

### Evaluating models
Once you have a collection of experimental trials contained in a folder, e.g. specified by ``log_path`` in your config file, we can evaluate all these trials and aggregate the results and do some evaluation. Note that within your ``log_path`` folder, each (seed, view) pair will have its own subfolder. For example, if you train models for 3 random seeds and 2 different scenes, you will have 6 subfolders in your logging path folder. To aggregate these results, you only need to edit four items in the ``singlescene_benchmark.py`` file.

First, you should specify the name of a folder for the ``save_path`` variable to save evaluation results. Make sure to create this folder before running this code. Second, specify the name of your logging path(s) you would like to evaluate. Referring to the previous example, if your logging path is named "MyLogs" and contains the 6 subfolders from training models on 2 scenes with three random seeds, you can just assign the ``folders`` variable as ``['MyLogs']``. This variable is a list in case you may have multiple distinct sets of experiments you want to evaluate separately. Next, specify the names of the scenes you would like evaluated with the ``views`` list. Lastly, ``n_train_list`` gives the number of training images you used for each (seed, view) pair. For example, if you trained all models with only 10 labeled images, then ``n_train_list=[10]``. If instead you tried 5 and 15 labeled images for each (seed, view) pair, then ``n_train_list=[5, 15]``.

Once these four items are set, you can compile all your evaluation results by running the following command:

``python3 singlescene_benchmark.py``

The evaluation results for each logging folder will be save in a separate JSON file within your ``save_path`` folder.

### Processing and examining results
The saved JSON file for each logging path folder contains a nested dictionary where we need to specify number of training images, training scene, and testing scene. For example, let ``results_dict`` be the loaded dictioanry from one such JSON file. If we assign ``curr_results = results_dict[10]['Almond at Washington East']['Hunt Club at Washington West']``, we will have a list of evaluation result dictionaries stored in ``curr_results``. The length of ``curr_results`` will correspond to the number of random seeds used to train models on the training scene, "Almond at Washington East" in this case, for the given number of training images, 10 in this case. Thus, if we have 3 random seeds, ``curr_results`` will be a length-3 list where ``curr_results[0]`` would give us one evaluation results dictionary for training on "Almond at Washington East" with 10 training images and then evaluating this one model on the "Hunt Club at Washington West" scene.

The evaluation results dictionary stored in ``curr_results[0]`` (or ``curr_results[1]`` and ``curr_results[2]`` as well) contain several evaluation metrics computed within the ``quantitative_evaluation.py`` file. Continuing this example, let ``curr_evaluation_dict = curr_results[0]``. The main metrics that are a good starting point are the F-measure, precision, and recall of each model. To get these values we can use the following lines of code:

``model_f_measure = curr_evaluation_dict['Total']['F-measure']``

``model_precision = curr_evaluation_dict['Total']['Precision']``

``model_recall = curr_evaluation_dict['Total']['Recall']``

There are many other available metrics in these dictionaries, but these should be a good starting point for evaluation. Finally, if we take the model's F-measure across the 3 random seeds in this example, we can find the average F-measure for training on "Almond at Washington East" with 10 labeled images and evaluating on "Hunt Club at Washington West". We may also find the standard deviation of the F-measure or also look at precision and recall, respectively.


## Overview of provided codes and files
A brief explanation of each provided file is given below.

### ``dataset.py``
### ``models.py``
### ``train.py``
### ``quantitative_evaluation.py``
### ``singlescene_benchmark.py``
### ``unet_singlescene_main.py``
### ``config.yaml``
