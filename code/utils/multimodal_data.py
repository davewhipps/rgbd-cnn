import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MultipleInputGenerator(tensorflow.keras.utils.Sequence):
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