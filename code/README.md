# Image Classifier Training and Testing

## Overview
This folder contains python code to train a CNN image classifier on the L-AVATeD Image Dataset. 

(NOTE: The L-AVATeD dataset must be present on your local machine. The default/expected location is in the `../data/` folder.)

Simple RGB or LiDAR training is initiated with `train.py` passing appropriate command line arguments. Several hyperparameters can be modified using the `hyperparameters/hyperparams_[rgb,lidar].yaml` files.

For example, to train a simple classifier on the RGB data:
```
> python train.py --data_dir "../data/lavated_rgb_split" --params_file "hyperparameters/hyperparams_rgb.yaml" --output_dir "../output"  
```

Simple RGB or LiDAR Model performance can be checked using the script `test.py` passing appropriate command line arguments.

For example, to test a model pre-trained on the LiDAR data:
```
> python test.py --model_dir "../models/lavated-lidar" --data_dir "../data/lavated_lidar_split" --output_dir "../output" 
```

Multimodal RGB-D (RGB and LiDAR) image pair model training is initiated with `train_rgbd.py`. Several hyperparameters can be modified using the `hyperparameters/hyperparams_rgbd.yaml` file.

```
> python train_rgbd.py --data_dir "../data" --params_file "hyperparameters/hyperparams_rgbd.yaml" --output_dir "../output"  
```

Simple RGB-D Model performance can be checked using the script `test_rgbd.py` passing appropriate command line arguments.

```
> python test_rgbd.py --model_dir "../models/lavated-rgbd" --data_dir "../data" --output_dir "../output" 
```

By default, model checkpoints, learning curve plots, and best models are written to an `output` folder with a date stamp.
