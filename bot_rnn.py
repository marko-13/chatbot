from tensorflow.keras import layers
import tensorflow as tf

from rnn_dataset import Dataset

class RNNModel():

    def __init__(self, dataset):


        self.model = tf.keras.Sequential()

        # TODO figure out how to count corpus from keras lib
        self.model.add(layers.Embedding())