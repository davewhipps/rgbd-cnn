import os
import datetime
import argparse
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight 
from collections import Counter
import pathlib
from utilities import add_prefix
import platform

#Parametrize hyperparams so we can grid search
HYPERPARAMS = {
  'WANTS_TRAIN_FULL_MODEL'  : True,
  'NUM_EPOCHS' : 20,
  'BATCH_SIZE' : 32,
  'IMG_SIZE' : (512, 384),
  'REGULARIZATION' : 0.5,
  'BASE_LEARNING_RATE' : 0.0001,
  'PATIENCE' : 10,
  'USE_CACHING' : False
}

# Main entry point
if __name__ == "__main__":
	# Get the command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, required=True )
	parser.add_argument('--output_dir', type=str, required=True )
	parser.add_argument('--params_file', type=str, required=False )
	args = parser.parse_args()

	# Store the output directory
	OUTPUT_DIR = args.output_dir

	# Read in hyperparams from YAML file, if any
	params_file = None
	if args.params_file:
		with open(args.params_file, 'r') as stream:
			try:
				params_file = yaml.load(stream, Loader=yaml.FullLoader)
				# Override any hyperparams with those in the YAML file
				for key, value in params_file.items():
					HYPERPARAMS[key] = value  
				print(HYPERPARAMS)

			except yaml.YAMLError as exc:
				print(exc)

	# Reality check
	print( "Using Tensorflow: ", tf.__version__)
	if tf.config.list_physical_devices('GPU'):
		print("TensorFlow **IS** using the GPU")
	else:
		print("TensorFlow **IS NOT** using the GPU")

	# rename for readability
	batch_size = HYPERPARAMS['BATCH_SIZE']
	image_size = HYPERPARAMS['IMG_SIZE']

	# Some data folder name constants
	data_source_folder = args.data_dir
	rgb_data_folder_name = "s2mnet_split"
	lidar_data_folder_name = "s2mnet_split_lidar"
	train_data_folder_name = "train"
	val_data_folder_name = "val"

	# Construct the paths to our training and validation data for RGB and Lidar images
	rgb_train_dir = os.path.join(data_source_folder, rgb_data_folder_name, train_data_folder_name ) #path to RGB training data
	rgb_valid_dir = os.path.join(data_source_folder, rgb_data_folder_name, val_data_folder_name ) #path to RGB validation data

	lidar_train_dir = os.path.join(data_source_folder, lidar_data_folder_name, train_data_folder_name ) #path to Lidar training data
	lidar_valid_dir = os.path.join(data_source_folder, lidar_data_folder_name, val_data_folder_name ) #path to Lidar validation data

	# Tease out the class names and number of classes
	class_names = np.array(sorted([item.name for item in pathlib.Path(rgb_train_dir).glob('*')]))
	num_classes = len(class_names)
	print(class_names, " Num Classes: ", num_classes) # Output the class names

	train_img_count = 0
	train_labels = []
	for class_name in class_names:
		DIR = os.path.join(rgb_train_dir, class_name)
		num_items = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		for i in range( num_items ):
			train_labels.append(class_name)
		train_img_count = train_img_count + num_items

	class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
	class_weights = dict(enumerate(class_weights)) 
	print("Class Weights: ", class_weights) # Output the class weights

	class MultipleInputGenerator(tf.keras.utils.Sequence):
		def __init__(self, dir_train_01, dir_train_02, batch_size):
			# Keras generator
			self.generator = ImageDataGenerator()

			# Real time multiple input data augmentation
			self.genX1 = self.generator.flow_from_directory(
				dir_train_01,
				target_size = HYPERPARAMS['IMG_SIZE'],
				batch_size  = HYPERPARAMS['BATCH_SIZE'],
				seed        = 42,
				shuffle		= True
			)
			self.genX2 = self.generator.flow_from_directory(
				dir_train_02,
				target_size = HYPERPARAMS['IMG_SIZE'],
				batch_size  = HYPERPARAMS['BATCH_SIZE'],
				seed        = 42,
				shuffle		= True
			)

		def __len__(self):
			"""It is mandatory to implement it on Keras Sequence"""
			return self.genX1.__len__()

		def __getitem__(self, index):
			"""Getting items from the 2 generators and packing them"""
			X1_batch, Y_batch = self.genX1.__getitem__(index)
			X2_batch, Y_batch = self.genX2.__getitem__(index)

			X_batch = [X1_batch, X2_batch]
			return X_batch, Y_batch

	# Read-in image pairs
	train_ds = MultipleInputGenerator( rgb_train_dir, lidar_train_dir, batch_size )
	val_ds = MultipleInputGenerator( rgb_valid_dir, lidar_valid_dir, batch_size )

	# Create the base model from the pre-trained model MobileNet V2
	RGB_IMG_SHAPE = image_size + (3,)
	print("Image Shape ", RGB_IMG_SHAPE)
	base_model_rgb = tf.keras.applications.MobileNetV2(	input_shape=RGB_IMG_SHAPE,
														include_top=False,
														weights='imagenet')
	base_model_rgb = add_prefix(base_model_rgb, 'rgb_') #so we don't run into unique layer naming error

	base_model_lidar = tf.keras.applications.MobileNetV2(input_shape=RGB_IMG_SHAPE,
														 include_top=False,
														 weights='imagenet')
	
	# We want to parametrize whether we want to train the entire model so we can test
	base_model_rgb.trainable = HYPERPARAMS['WANTS_TRAIN_FULL_MODEL']
	base_model_lidar.trainable = HYPERPARAMS['WANTS_TRAIN_FULL_MODEL']

	# Define RGB image pipeline
	rgb_input = tf.keras.Input(shape=RGB_IMG_SHAPE, batch_size=batch_size)
	# Our dataset is somewhat limited, so do some data augmentation to beef it up
	rgb_pipeline = tf.keras.layers.RandomRotation(0.05)(rgb_input)
	# We're using MobileNetV2, so use the built-in preprocessing for that base model
	rgb_pipeline = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_pipeline) #rescales from -1..1 for MobileNetV2
	rgb_pipeline = base_model_rgb(rgb_pipeline, training=False)
	rgb_pipeline = tf.keras.layers.GlobalAveragePooling2D()(rgb_pipeline)
	rgb_pipeline = tf.keras.layers.Dropout( HYPERPARAMS['REGULARIZATION'] )(rgb_pipeline) # regularize for better generalization

	# Define Lidar image pipeline
	lidar_input = tf.keras.Input(shape=RGB_IMG_SHAPE, batch_size=batch_size)
	# Our dataset is somewhat limited, so do some data augmentation to beef it up
	lidar_pipeline = tf.keras.layers.RandomRotation(0.05)(lidar_input)
	# We're using MobileNetV2, so use the built-in preprocessing for that base model
	lidar_pipeline = tf.keras.applications.mobilenet_v2.preprocess_input(lidar_pipeline) #rescales from -1..1 for MobileNetV2
	lidar_pipeline = base_model_lidar(lidar_pipeline, training=False)
	lidar_pipeline = tf.keras.layers.GlobalAveragePooling2D()(lidar_pipeline)
	lidar_pipeline = tf.keras.layers.Dropout( HYPERPARAMS['REGULARIZATION'] )(lidar_pipeline) # regularize for better generalization
	
	# Concatenate RGB and Lidar output
	conv = keras.layers.concatenate([rgb_pipeline, lidar_pipeline])
	conv = keras.layers.Flatten()(conv)
	output = keras.layers.Dense(num_classes, activation='softmax')(conv)

	# Define the full model taking in two images as input and a single image class as output
	model = tf.keras.Model(inputs=[rgb_input, lidar_input], outputs=output)

	# print a summary of our full model
	model.summary()

	if platform.processor() == 'arm':
		optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=HYPERPARAMS['BASE_LEARNING_RATE'])
	else:
		optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS['BASE_LEARNING_RATE'])

	#Compile the model
	model.compile(	optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(),
					weighted_metrics = ['accuracy'] )

	# Use checkpointing so that we can resume, as well as test and export
	date_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", date_time_string, "cp-{epoch:04d}.ckpt")
	checkpoint_dir = os.path.dirname(checkpoint_path)
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
							  								 save_best_only=True,
                                                             verbose=1)

	# Use early stopping so we don't need to worry too much about our NUM_EPOCHS
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=HYPERPARAMS['PATIENCE'])

	# Train
	
	history = model.fit(train_ds,
						validation_data=val_ds,
						epochs=HYPERPARAMS['NUM_EPOCHS'],
						class_weight=class_weights,
						callbacks=[checkpoint_callback, early_stopping])

	# Save out the "best" model
	final_model_path = os.path.join(OUTPUT_DIR, "saved_models", date_time_string)
	os.makedirs(final_model_path, exist_ok = True)
	model.save(final_model_path+'/s2mnet_model')

	# Plot learning curves, and save as an image
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

	learning_curves_file = os.path.join(OUTPUT_DIR, "plots", date_time_string, "learning_curves.png")
	training_plots_dir = os.path.dirname(learning_curves_file)
	if not os.path.isdir(training_plots_dir):
		os.makedirs(training_plots_dir)
	plt.savefig(learning_curves_file, bbox_inches='tight')