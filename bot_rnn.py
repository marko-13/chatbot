from tensorflow.keras import layers
import tensorflow as tf
import joblib

from rnn_dataset import Dataset

class RNNModel():

    '''
    TODO:
        1. Napisati funkciju za trening modela (koristi se rnn_dataset.Dataset klasa, 
            tamo se nalaze sve utility funkcije za izvlacenje originalnih stringova pitanja
            i parafraziranih pitanja, kao i iteriranje kroz trening skup slicnih i ne slicnih
            parova pitanja)
        2. Napisati funkcije za snimanje i ucitavanje modela sa diska
        3. Napisati funkciju koja prima string (input od korisnika), iskorisit high recall
            algoritam za pronalazenje top 100 slicnih pitanja, pa iz njih izvlaci najbolje
            pomocu RNN-a. (Pogledati bot.py fajl, funkcija 'process input')
    '''

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
