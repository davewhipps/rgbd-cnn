# S2M Image Classifier Training and Testing

## Overview

This folder contains python code to train a CNN image classifier on the S2Mnet Image Dataset, the location of which must be pre-separated into train, validation and test subfolders (easily accomplished using `s2mnet_data_prep/copy_and_split_data.py`).

Model training is initiated with the `train_classifer.sh` script. Several hyperparameters can be modified using the hyperparameters.yaml file.

Model performance can be checked using the script `test_classifier.sh` which calls `test.py` with several hyperparameters.

By default, model checkpoints, learning curve plots, and best models are written to an `output` folder with a date stamp.
