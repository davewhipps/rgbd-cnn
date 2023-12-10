import os
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
import pathlib
import numpy as np
from sklearn.utils import class_weight 

def plot_history( history, output_dir, date_time_string ):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    learning_curves_file = os.path.join(output_dir, "plots", date_time_string, "learning_curves.png")
    training_plots_dir = os.path.dirname(learning_curves_file)
    if not os.path.isdir(training_plots_dir):
        os.makedirs(training_plots_dir)
    plt.savefig(learning_curves_file, bbox_inches='tight')

def read_hyperparameters( params_file, default_hyperparams = {} ):
    HYPERPARAMS = default_hyperparams
    if params_file:
        with open(params_file, 'r') as stream:
            try:
                params_file = yaml.load(stream, Loader=yaml.FullLoader)
                # Override any hyperparams with those in the YAML file
                for key, value in params_file.items():
                    HYPERPARAMS[key] = value  
                print(HYPERPARAMS)

            except yaml.YAMLError as exc:
                print(exc)
    return HYPERPARAMS  

def get_class_weights( image_file_train_dir_path ):
    # Tease out the class names and number of classes
    image_file_train_dir = pathlib.Path(image_file_train_dir_path)
    if not image_file_train_dir.exists():
        print('Directory does not exist: ', image_file_train_dir_path)
        raise Exception('Directory does not exist: '+image_file_train_dir_path)
        return
    
    class_names = np.array(sorted([item.name for item in image_file_train_dir.glob('*') if not item.name.startswith('.')]))
    
    num_classes = len(class_names)
    print(class_names, " Num Classes: ", num_classes) # Output the class names

    train_img_count = 0
    train_labels = []
    for class_name in class_names:
        DIR = os.path.join(image_file_train_dir_path, class_name)
        num_items = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        for i in range( num_items ):
            train_labels.append(class_name)
        train_img_count = train_img_count + num_items

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))
    return ( class_weights, num_classes )