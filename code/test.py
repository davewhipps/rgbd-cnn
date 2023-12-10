import os
import math
import datetime
import argparse
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sn
import pandas as pd

# Parametrize hyperparams so we can grid search
HYPERPARAMS = {
    'BATCH_SIZE': 32,
    'IMG_SIZE': (256, 192)
}

# Main entry point
if __name__ == "__main__":
    # Get the command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Store the output directory
    OUTPUT_DIR = args.output_dir

    # Read in the test data
    test_data_path = os.path.join(args.data_dir, 'test')
    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        test_data_path,
        target_size=HYPERPARAMS['IMG_SIZE'],
        batch_size=HYPERPARAMS['BATCH_SIZE'],
        shuffle=False)
    test_steps_per_epoch = math.ceil(
        test_data_generator.samples / test_data_generator.batch_size)

    # load the model
    model = tf.keras.models.load_model(args.model_dir)

    predictions = model.predict(
        test_data_generator, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = metrics.classification_report(
        true_classes, predicted_classes, target_names=class_labels)
    print(report)
    report_dict = metrics.classification_report(
        true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    # So we can output to a file
    report_df = pd.DataFrame(report_dict).transpose()

    confusion_matrix = confusion_matrix(
        true_classes, predicted_classes, normalize='pred')
    print(confusion_matrix)

    # Pretty plot the confusion matrix
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    sn.set(font_scale=1.1)  # for label size
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='.1%',
                    annot_kws={"size": 10}, cbar=False, cmap='Purples')
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax1.set_ylabel('True Values', fontsize=14)
    ax1.set_xlabel('Predicted Values', fontsize=14)

    date_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    confusion_matrix_file = os.path.join(
        OUTPUT_DIR, date_time_string, "confusion_matrix.png")
    confusion_matrix_dir = os.path.dirname(confusion_matrix_file)
    if not os.path.isdir(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)
    plt.savefig(confusion_matrix_file, bbox_inches='tight')

    classification_report_file = os.path.join(
        OUTPUT_DIR, date_time_string, "classification_report.txt")
    with open(classification_report_file, 'w') as outfile:
        report_df.to_string(outfile)
