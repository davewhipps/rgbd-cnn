import tensorflow as tf

def get_base_model( image_shape, batch_size, regularization, layer_prefix ):
    """This method creates the basic model architecture used throughout our experiments.

    Parameters:
    image_shape ( Tuple ): A tuple defining the (width, height, depth) of the image. (Depth required to be 3, for MobileNetV2 )
    batch_size ( Number ): A number indicating the batch size
    regularization ( Number ): A number indicating the regularization parameter [0..1]
    layer_prefix ( String ): A string to prepend to sub-layers in the MobileNetV2 instance (preventing name colissions when multiple instances)

    Returns:
    tuple ( tf.keras.Input, tf.keras.Model ): A tuple whose first object is a reference to the input layer,
    and whose second object is the full model itself
    """
    # Input layer
    input_layer = tf.keras.Input(shape=image_shape, batch_size=batch_size)

    # Our dataset is somewhat limited, so do some data augmentation to beef it up
    model = tf.keras.layers.RandomRotation(0.05)(input_layer)
    
    # We're using MobileNetV2, so use the built-in preprocessing for that base model
    model = tf.keras.applications.mobilenet_v2.preprocess_input(model) #rescales from -1..1 for MobileNetV2

    # Instantiate MobileNetV2, imagenet weights are just for initialization, as we wil train the entire model
    base_model = tf.keras.applications.MobileNetV2(	input_shape=image_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    
    # This prepends our "layer_prefix" to the MobileNetV2 sub-layers so we don't run into unique layer naming error
    base_model = add_prefix_to_model(base_model, layer_prefix )

    # We want to use the full model capacity here and allow all layers to train on our data
    base_model.trainable = True

    model = base_model(model, training=False)

    # Add pooling (could probably also pass pooling="avg" into the instantiation above)
    model = tf.keras.layers.GlobalAveragePooling2D()(model)

    # regularize for better generalization    
    model = tf.keras.layers.Dropout( regularization )(model)

    return (input_layer, model)

# Because we're occasionally using two instances of the MobileNetV2 model in our pipeline, we need
# to rename some of the layers, so that we don't have name colissions in our combined
# architecture
def add_prefix_to_model(model, prefix: str, custom_objects=None):
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
