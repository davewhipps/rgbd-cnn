import os
import time
import math
import datetime
import argparse
import tensorflow as tf
import numpy as np
from utilities import MultipleInputGenerator

HYPERPARAMS = {
    'BATCH_SIZE': 32,
    'IMG_SIZE': (256, 192)
}

# Main entry point
if __name__ == "__main__":
    # Get the command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # Read in the test data
    data_source_folder = args.data_dir
    rgb_data_folder_name = "s2mnet_data_split"
    lidar_data_folder_name = "s2mnet_data_split_lidar"
    test_data_folder_name = "test"

    rgb_test_dir = os.path.join(
        data_source_folder, rgb_data_folder_name, test_data_folder_name)  # path to RGB test data
    lidar_test_dir = os.path.join(
        data_source_folder, lidar_data_folder_name, test_data_folder_name)  # path to Lidar test data

    test_data_generator = MultipleInputGenerator(
        rgb_test_dir, lidar_test_dir, HYPERPARAMS['BATCH_SIZE'], HYPERPARAMS['IMG_SIZE'], shuffle=False)
    test_steps_per_epoch = math.ceil(
        test_data_generator.samples / test_data_generator.batch_size)

    # load the model
    model = tf.keras.models.load_model(args.model_dir)

    # We have 800 test images
    WARMUP_BATCHES = int(5)
    TOTAL_BATCHES = int(800 / HYPERPARAMS['BATCH_SIZE'])
    TEST_BATCHES = int(TOTAL_BATCHES - WARMUP_BATCHES)

    test_data = iter(test_data_generator)
    # Run a few iterations to make sure everything is loaded into memory
    for _ in range(WARMUP_BATCHES - 1):
        images, labels = next(test_data)
        model.predict(images, verbose=0)

    times = []

    for _ in range(TEST_BATCHES - 1):
        images, labels = next(test_data)
        start_time = time.time()
        model.predict(images, verbose=0)
        end_time = time.time()
        # We want per-sample, not per-batch
        times.append(1000*(end_time - start_time) / HYPERPARAMS['BATCH_SIZE'])

    print(f"Model       : {model.name}")
    print(f"Batches     : {TEST_BATCHES}")
    print(f"Average (ms): {np.mean(times)}")
    print(f"Std dev (ms): {np.std(times)}")
    print(f"Min (ms)    : {np.min(times)}")
    print(f"Max (ms)    : {np.max(times)}")
