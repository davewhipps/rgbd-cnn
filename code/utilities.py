import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml
import pathlib
import numpy as np
from sklearn.utils import class_weight 


class MultipleInputGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_train_01, dir_train_02, batch_size, image_size, shuffle=True):
        # Keras generator
        self.generator = ImageDataGenerator()

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow_from_directory(
            dir_train_01,
            target_size = image_size,
            batch_size  = batch_size,
            seed        = 42,
            shuffle		= shuffle
        )
        self.genX2 = self.generator.flow_from_directory(
            dir_train_02,
            target_size = image_size,
            batch_size  = batch_size,
            seed        = 42,
            shuffle		= shuffle
        )

        self.samples = self.genX1.samples
        self.batch_size = self.genX1.batch_size
        self.classes = self.genX1.classes
        self.class_indices = self.genX1.class_indices
        
    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)

        X_batch = [X1_batch, X2_batch]
        return X_batch, Y_batch
          
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


# Because we're using two instances of the MobileNetV2 model in our pipeline, we need
# to rename some of the layers, so that we don't have name colissions in our combined
# architecture
def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''

    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    return new_model

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

def get_class_weights( image_file_train_dir ):
    # Tease out the class names and number of classes
    class_names = np.array(sorted([item.name for item in pathlib.Path(image_file_train_dir).glob('*') if not item.name.startswith('.')]))
    
    num_classes = len(class_names)
    print(class_names, " Num Classes: ", num_classes) # Output the class names

    train_img_count = 0
    train_labels = []
    for class_name in class_names:
        DIR = os.path.join(image_file_train_dir, class_name)
        num_items = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        for i in range( num_items ):
            train_labels.append(class_name)
        train_img_count = train_img_count + num_items

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))
    return ( class_weights, num_classes )
	
def get_base_pipeline( image_shape, batch_size, regularization, layer_prefix ):
    # Define image pipeline
    input_layer = tf.keras.Input(shape=image_shape, batch_size=batch_size)
    # Our dataset is somewhat limited, so do some data augmentation to beef it up
    pipeline = tf.keras.layers.RandomRotation(0.05)(input_layer)
    # We're using MobileNetV2, so use the built-in preprocessing for that base model
    pipeline = tf.keras.applications.mobilenet_v2.preprocess_input(pipeline) #rescales from -1..1 for MobileNetV2

    base_model = tf.keras.applications.MobileNetV2(	input_shape=image_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    
    base_model = add_prefix(base_model, layer_prefix ) #so we don't run into unique layer naming error
    base_model.trainable = True
    pipeline = base_model(pipeline, training=False)

    pipeline = tf.keras.layers.GlobalAveragePooling2D()(pipeline)
    pipeline = tf.keras.layers.Dropout( regularization )(pipeline) # regularize for better generalization    
    return (input_layer, pipeline)
