from gensim.models import KeyedVectors
from tensorflow.keras import layers
import tensorflow as tf
import joblib
import pickle
import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.models import model_from_json
from rnn_dataset import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    def __init__(self, list_of_pairs, pair_y):


        # find all disinct words(tokens) from original questions
        token_dict = {}
        inv_token_dict = {}
        all_tokens = []
        brojac = 1
        for question in Dataset.get_original_dataset_questions(load_object("objects/rnn_dataset_object.pickle")):
            tokens = tf.keras.preprocessing.text.text_to_word_sequence(question,
                                                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                                       lower=True, split=' ')
            for t in tokens:
                if t not in all_tokens:
                    token_dict[brojac] = t
                    inv_token_dict[t] = brojac
                    brojac += 1
                    # print(t)
                    all_tokens.append(t)



        self.model = tf.keras.Sequential()
        self.model.add(layers.Embedding(input_dim=len(all_tokens)+1, output_dim=100))
        self.model.add(layers.LSTM(128))
        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        X_train_para = []
        X_train_orig = []
        for pairs in list_of_pairs:
            X_train_para.append(split_and_zero_padding(pairs[0], 15, token_dict, inv_token_dict))
            X_train_orig.append(split_and_zero_padding(pairs[1], 15, token_dict, inv_token_dict))


        # Start training
        # TODO
        # jos treba istrenirati mrezu, podaci su spremni, padding odradjen, word embedding isto, sve je stavljeno
        # u liste odgovarajuce duzine
        ma_lstm_trained = self.model.fit(X_train_orig, X_train_para, pair_y, batch_size=32, epochs=50)


        # self.model.save('./objects/SiameaseLSTM.h5')

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


def split_and_zero_padding(quesion_paraquestion, max_seq_len, token_dict, inv_token_dict):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(quesion_paraquestion,
                                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                               lower=True, split=' ')
    # print(quesion_paraquestion)
    word_embedding_matrix = []
    for word in tokens:
        if word in token_dict.values():
            word_embedding_matrix.append(inv_token_dict[word])
            # print(word)
            # print(inv_token_dict[word])

        # TODO
        # ako ne postoji rec u dict, sta onda, da stavi nulu ili da doda u recnik sa novom verdnoscu

    # word_embedding_matrix = np.asarray(word_embedding_matrix, dtype=np.float32)
    word_embedding_matrix =([word_embedding_matrix])
    ret_val = pad_sequences(word_embedding_matrix, padding='pre', truncating='post', maxlen=max_seq_len)

    # embedding za recenicu
    print(word_embedding_matrix)
    # embeddings za recenicu nakon paddinga
    print(ret_val)
    return ret_val

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))