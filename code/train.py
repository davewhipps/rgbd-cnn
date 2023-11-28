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

#Parametrize hyperparams so we can grid search
HYPERPARAMS = {
  'WANTS_TRAIN_FULL_MODEL'  : True,
  'NUM_EPOCHS' : 20,
  'BATCH_SIZE' : 32,
  'IMG_SIZE' : (512, 384),
  'REGULARIZATION' : 0.5,
  'BASE_LEARNING_RATE' : 0.0001,
  'USE_CACHING' : False,
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

	# Read in the training data
	train_dir = os.path.join(args.data_dir, 'train')
	train_data = tf.keras.utils.image_dataset_from_directory(train_dir,
															seed=42,
															shuffle=True,
															batch_size=HYPERPARAMS['BATCH_SIZE'],
															image_size=HYPERPARAMS['IMG_SIZE'])

	# Read in the validation data
	validation_dir = os.path.join(args.data_dir, 'val')
	val_data = tf.keras.utils.image_dataset_from_directory(validation_dir,
															seed=42,
															shuffle=True,
															batch_size=HYPERPARAMS['BATCH_SIZE'],
															image_size=HYPERPARAMS['IMG_SIZE'])

	# Tease out the class names
	CLASS_NAMES = train_data.class_names
	NUM_CLASSES = len(CLASS_NAMES)
	print(CLASS_NAMES, " Num Classes: ", NUM_CLASSES) # Output the class names

	# Prefetch for performance
	AUTOTUNE = tf.data.AUTOTUNE
	train_data = train_data.prefetch(buffer_size=AUTOTUNE)
	val_data = val_data.prefetch(buffer_size=AUTOTUNE)

	# Our dataset is imbalanced, so calculate the class weights
	train_labels = np.concatenate([y for x, y in train_data], axis=0)
	class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
	class_weights = dict(enumerate(class_weights)) 
	print("Class Weights: ", class_weights) # Output the class weights

	# Create the base model from the pre-trained model MobileNet V2
	IMG_SHAPE = HYPERPARAMS['IMG_SIZE'] + (3,)
	print("Image Shape ", IMG_SHAPE)
	base_model = tf.keras.applications.MobileNetV2(	input_shape=IMG_SHAPE,
													include_top=False,
													weights='imagenet')

	# We want to parametrize whether we want to train the entire model so we can test
	base_model.trainable = HYPERPARAMS['WANTS_TRAIN_FULL_MODEL']

	# Define pipeline
	inputs = tf.keras.Input(shape=IMG_SHAPE)
	# Our dataset is somewhat limited, so do some data augmentation to beef it up
	x = tf.keras.layers.RandomRotation(0.05)(inputs) 
	# We're using MobileNetV2, so use the built-in preprocessing for that base model
	x = tf.keras.applications.mobilenet_v2.preprocess_input(x) #rescales from -1..1 for MobileNetV2
	# Call MobileNetV2
	x = base_model(x, training=False)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(HYPERPARAMS['REGULARIZATION'])(x) # regularize for better generalization
	outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	# print a summary of our full model
	model.summary()

	#Compile the model
	# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs
	# please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
	if platform.processor() == 'arm':
		optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=HYPERPARAMS['BASE_LEARNING_RATE'])
	else:
		optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS['BASE_LEARNING_RATE'])

	model.compile(	optimizer=optimizer,
					loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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
	history = model.fit(train_data,
						validation_data=val_data,
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
