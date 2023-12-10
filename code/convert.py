import argparse
import coremltools as ct 
import tensorflow as tf

# Main entry point
if __name__ == "__main__":

	# Get the command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', type=str, required=True )
	parser.add_argument('--labels_file', type=str, required=True )
	parser.add_argument('--output_file', type=str, required=True )
	args = parser.parse_args()
	
	# Load the model
	tf_model = tf.keras.models.load_model(args.model_dir)

	# Add some configuration
	config = ct.ClassifierConfig(args.labels_file)

	# Define the input image
	image_input = ct.ImageType(name="input_2", shape=(1, 224, 224, 3,))

	# Convert the model to CoreML package
	model = ct.convert(tf_model, inputs=[image_input], classifier_config=config, convert_to="mlprogram")

	# Add some Metadata	
	model.author = 'David Whipps'
	model.license = 'MIT'
	model.short_description = 'Predicts Walking Terrain'
	model.version = '1'

	# Set feature descriptions manually
	model.input_description['input_2'] = 'RGB Image of the current scene'

	# Set the output descriptions
	model.output_description['classLabel'] = 'Walking Terrain Type'
	model.output_description['classLabel_probs'] = 'Output Probabilities'

	#write out the model
	model.save(args.output_file)