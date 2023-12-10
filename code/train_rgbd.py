""" This module trains a multimodal, parallel architecture model on the L-AVATeD dataset
"""

import os
import datetime
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import Counter
import pathlib
from utils.models import get_base_model
from utils.utilities import plot_history, read_hyperparameters, get_class_weights
from utils.multimodal_data import MultipleInputGenerator
import platform

#Default hyperparams
HYPERPARAMS = {
  'NUM_EPOCHS' : 20,
  'BATCH_SIZE' : 32,
  'IMG_SIZE' : (512, 384),
  'REGULARIZATION' : 0.5,
  'BASE_LEARNING_RATE' : 0.0001,
  'PATIENCE' : 10
}

# Main entry point
if __name__ == "__main__":

	# Get the command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, required=False, default="../data" )
	parser.add_argument('--output_dir', type=str, required=False, default="../output/rgbd" )
	parser.add_argument('--params_file', type=str, required=False, default="hyperparameters/hyperparams_rgbd.yaml" )
	args = parser.parse_args()

	# Ensure we're working with absolute paths
	data_dir = os.path.abspath(args.data_dir)
	output_dir = os.path.abspath(args.output_dir)

	# Read in hyperparams from YAML file, if any
	if args.params_file:
		params_file = os.path.abspath(args.params_file)
		HYPERPARAMS = read_hyperparameters( params_file, HYPERPARAMS )

	# rename for readability
	batch_size = HYPERPARAMS['BATCH_SIZE']
	image_size = HYPERPARAMS['IMG_SIZE']
	regularization = HYPERPARAMS['REGULARIZATION']
	learning_rate = HYPERPARAMS['BASE_LEARNING_RATE']
	patience = HYPERPARAMS['PATIENCE']
	epochs = HYPERPARAMS['NUM_EPOCHS']

	# Some data folder name constants
	data_source_folder = data_dir
	rgb_data_folder_name = "lavated_data_split_rgb"
	lidar_data_folder_name = "lavated_data_split_lidar"
	train_data_folder_name = "train"
	val_data_folder_name = "val"

	# Construct the paths to our training and validation data for RGB and Lidar images
	rgb_train_dir = os.path.join( data_source_folder, rgb_data_folder_name, train_data_folder_name ) #path to RGB training data
	rgb_valid_dir = os.path.join( data_source_folder, rgb_data_folder_name, val_data_folder_name ) #path to RGB validation data

	lidar_train_dir = os.path.join( data_source_folder, lidar_data_folder_name, train_data_folder_name ) #path to Lidar training data
	lidar_valid_dir = os.path.join( data_source_folder, lidar_data_folder_name, val_data_folder_name ) #path to Lidar validation data

	# The L-AVATeD dataset has unbalanced classes, and so we need to account for that by weighting our loss
	class_weights, num_classes = get_class_weights( rgb_train_dir  )
	print("Class Weights: ", class_weights) # Output the class weights

	# Create Image Generators for image pairs of RGB and LiDAR data
	train_ds = MultipleInputGenerator( rgb_train_dir, lidar_train_dir, batch_size, image_size )
	val_ds = MultipleInputGenerator( rgb_valid_dir, lidar_valid_dir, batch_size, image_size )

	# Create the base model from the pre-trained model MobileNet V2
	image_shape = image_size + (3,)
	print("Image Shape ", image_shape)
	( rgb_input,   rgb_pipeline )   = get_base_model( image_shape, batch_size, regularization, 'rgb_')
	( lidar_input, lidar_pipeline ) = get_base_model( image_shape, batch_size, regularization, 'lidar_')

	# Concatenate RGB and Lidar output
	conv = keras.layers.concatenate([rgb_pipeline, lidar_pipeline])
	conv = keras.layers.Flatten()(conv)
	output = keras.layers.Dense(num_classes, activation='softmax')(conv)

	# Define the full model taking in two images as input and a single image class as output
	model = tf.keras.Model(inputs=[rgb_input, lidar_input], outputs=output)

	# print a summary of our full model
	model.summary()

	# If you're running on an ARM (aka Apple Silicon) processor, this will speed up training
	if platform.processor() == 'arm':
		optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
	else:
		optimizer = tf.keras.optimizers.Adam(learning_rate)

	#Compile the model
	model.compile(	optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(),
					weighted_metrics = ['accuracy'] )

	# Use checkpointing so that we can resume, as well as test and export
	date_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpoint_path = os.path.join(output_dir, "checkpoints", date_time_string, "cp-{epoch:04d}.ckpt")
	checkpoint_dir = os.path.dirname(checkpoint_path)
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
							  								 save_best_only=True,
                                                             verbose=1)

	# Use early stopping so we don't need to worry too much about our NUM_EPOCHS
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

	# Train
	history = model.fit(train_ds,
						validation_data=val_ds,
						epochs=epochs,
						class_weight=class_weights,
						callbacks=[checkpoint_callback, early_stopping])

	# Save out the "best" model
	final_model_path = os.path.join(output_dir, "saved_models", date_time_string)
	os.makedirs(final_model_path, exist_ok = True)
	model.save(final_model_path+'/lavated-rgbd')

	# Plot learning curves, and save as an image
	plot_history( history, output_dir, date_time_string )