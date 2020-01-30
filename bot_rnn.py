from tensorflow.keras import layers
import tensorflow as tf
import joblib

from rnn_dataset import Dataset

class RNNModel():

    def __init__(self, dataset):



        self.model = tf.keras.Sequential()

                
        # TODO: figure out how to count the corpus words
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(dataset, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                                                       split=' ')


        self.model.add(layers.Embedding(len(tokens)))

        self.model.add(layers.LSTM(128))

        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

    def save_object(object_to_save, filename):
        with open(filename, 'wb') as output:
            joblib.dump(object_to_save, output)

    def load_object(filename):
        with open(filename, 'rb') as input_file:
            return joblib.load(input_file)
