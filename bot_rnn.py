from tensorflow.keras import layers
import tensorflow as tf
import joblib
import pickle

from tensorflow.keras.models import model_from_json
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

    def __init__(self):



        self.model = tf.keras.Sequential()

        all_tokens = []
        for question in Dataset.get_original_dataset_questions(load_object("objects/rnn_dataset_object.pickle")):
            tokens = tf.keras.preprocessing.text.text_to_word_sequence(question, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                                                       split=' ')
            for t in tokens:
                if t not in all_tokens:
                    print(t)
                    all_tokens.append(t)



        self.model.add(layers.Embedding(input_dim=len(all_tokens)+1, output_dim=100))

        self.model.add(layers.LSTM(128))

        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # PRVO URADITI FIT PA SACUVATI MODEL

    def serialize_ann(self, rnn):
            # serijalizuj arhitekturu neuronske mreze u JSON fajl
            model_json = rnn.to_json()
            with open("serialization_folder/neuronska.json", "w") as json_file:
                    json_file.write(model_json)
            # serijalizuj tezine u HDF5 fajl
            rnn.save_weights("serialization_folder/neuronska.h5")

    def load_trained_ann(self):
            try:
                    # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
                    json_file = open('serialization_folder/neuronska.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    ann = model_from_json(loaded_model_json)
                    # ucitaj tezine u prethodno kreirani model
                    ann.load_weights("serialization_folder/neuronska.h5")
                    print("Istrenirani model uspesno ucitan.")
                    return ann
            except Exception as e:
                    # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
                    return None


def load_object(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)