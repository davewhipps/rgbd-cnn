import os
import datetime
import argparse
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from collections import Counter
import platform
from utilities import plot_history, read_hyperparameters, get_class_weights, get_base_pipeline

# Parametrize hyperparams so we can grid search
HYPERPARAMS = {
    'WANTS_TRAIN_FULL_MODEL': True,
    'NUM_EPOCHS': 20,
    'BATCH_SIZE': 32,
    'IMG_SIZE': (512, 384),
    'REGULARIZATION': 0.5,
    'BASE_LEARNING_RATE': 0.0001,
    'USE_CACHING': False
}

# Main entry point
if __name__ == "__main__":
    # Get the command line args

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--params_file', type=str, required=False)
    args = parser.parse_args()

    # Store the output directory
    outpur_dir = args.output_dir

    # Read in hyperparams from YAML file, if any
    if args.params_file:
        HYPERPARAMS = read_hyperparameters(args.params_file, HYPERPARAMS)

    # rename for readability
    batch_size = HYPERPARAMS['BATCH_SIZE']
    image_size = HYPERPARAMS['IMG_SIZE']
    regularization = HYPERPARAMS['REGULARIZATION']
    learning_rate = HYPERPARAMS['BASE_LEARNING_RATE']
    patience = HYPERPARAMS['PATIENCE']
    epochs = HYPERPARAMS['NUM_EPOCHS']
    modality = HYPERPARAMS['MODALITY']

    # Read in the training data
    train_dir = os.path.join(args.data_dir, 'train')
    train_data = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                             seed=42,
                                                             shuffle=True,
                                                             batch_size=batch_size,
                                                             image_size=image_size)

    # Read in the validation data
    validation_dir = os.path.join(args.data_dir, 'val')
    val_data = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                           seed=42,
                                                           shuffle=True,
                                                           batch_size=batch_size,
                                                           image_size=image_size)

    # Prefetch for performance
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    # The L-AVATeD dataset has unbalanced classes, and so we need to account for that
    # by weighting our loss
    class_weights, num_classes = get_class_weights(train_dir)
    print("Class Weights: ", class_weights)  # Output the class weights

    # Create the base model from the pre-trained model MobileNet V2
    image_shape = image_size + (3,)
    print("Image Shape ", image_shape)
    (inputs, pipeline) = get_base_pipeline(
        image_shape, batch_size, regularization, modality+'_')
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax')(pipeline)
    model = tf.keras.Model(inputs, outputs)

    # print a summary of our full model
    model.summary()

    # If you're running on an ARM (aka Apple Silicon) processor, this will speed up training
    if platform.processor() == 'arm':
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  weighted_metrics=['accuracy'])

    # Use checkpointing so that we can resume, as well as test and export
    date_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(
        outpur_dir, "checkpoints", date_time_string, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                             save_best_only=True,
                                                             verbose=1)

    # Use early stopping so we don't need to worry too much about our NUM_EPOCHS
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience)

    # Train
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        class_weight=class_weights,
                        callbacks=[checkpoint_callback, early_stopping])

    # Save out the "best" model
    final_model_path = os.path.join(
        outpur_dir, "saved_models", date_time_string)
    os.makedirs(final_model_path, exist_ok=True)
    model.save(final_model_path+'/s2mnet_model')

    # Plot learning curves, and save as an image
    plot_history(history, outpur_dir, date_time_string)
