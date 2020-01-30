from tensorflow.keras import layers
import tensorflow as tf

from rnn_dataset import Dataset

class RNNModel():

    def __init__(self, dataset):
        


        self.model = tf.keras.Sequential()

        # TODO: figure out how to count the corpus words
        # self.model.add(layers.Embedding())

        self.model.add(layers.LSTM(128))

        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
