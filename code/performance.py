import os
import time
import math
import datetime
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


HYPERPARAMS = {
    'BATCH_SIZE': 32,
    'IMG_SIZE': (256, 192),
    'USE_CACHING': True
}

# Main entry point
if __name__ == "__main__":
    # Get the command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # load the model
    model = tf.keras.models.load_model(args.model_dir)

    input_sig = [tf.TensorSpec([1, *inputs.shape[1:]])
                 for inputs in model.inputs]
    forward_pass = tf.function(model.call, input_signature=input_sig)
    graph_info = profile(forward_pass.get_concrete_function(
    ).graph, options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate (MACCs)
    # as two flops
    maccs = graph_info.total_float_ops
    flops = maccs // 2
    print('MACCs: {:,}'.format(maccs))
    print('FLOPs: {:,}'.format(flops))

    # Read in the test data
    test_data_path = os.path.join(args.data_dir, 'test')
    test_data = tf.keras.utils.image_dataset_from_directory(test_data_path,
                                                            seed=42,
                                                            shuffle=True,
                                                            batch_size=HYPERPARAMS['BATCH_SIZE'],
                                                            image_size=HYPERPARAMS['IMG_SIZE'])
    test_data = iter(test_data)

    # We have 800 test images
    WARMUP_BATCHES = int(5)
    TOTAL_BATCHES = int(800 / HYPERPARAMS['BATCH_SIZE'])
    TEST_BATCHES = int(TOTAL_BATCHES - WARMUP_BATCHES)

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
