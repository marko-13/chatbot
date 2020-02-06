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

from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

from tensorflow import Tensor

import os

# BERT
from transformers import *


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

    def __init__(self, list_of_pairs, pair_y, train=False, bert=False):

        self.bert = bert

        if not bert:
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
                        all_tokens.append(t)

            self.token_dict = token_dict
            self.inv_token_dict = inv_token_dict

            X_train_para = []
            X_train_orig = []
            for pairs in list_of_pairs:
                X_train_para.append(split_and_zero_padding(pairs[0], 15, token_dict, inv_token_dict))
                X_train_orig.append(split_and_zero_padding(pairs[1], 15, token_dict, inv_token_dict))

            # DEFINE THE EMBEDDING - use the GloVe encoding

            # DICT
            self.glove_rep = self._load_glove()
            # self.embeddings = self._load_glove()

            # GloVe encoding
            self.embedding_dim = len(self.glove_rep['if'])

            # Convert the dict to a numpy matrix
            self.embeddings = np.zeros((len(all_tokens) + 1, self.embedding_dim))

            # i = 1
            # # Word embeddings for the entire corpus
            for index, word in token_dict.items():
                try:
                    vec = self.glove_rep[word]
                    self.embeddings[index] = vec
                except KeyError:

                    pass

            if train:
                self.train_model(list_of_pairs, pair_y, X_train_orig, X_train_para)
            else:
                self.load_model()

        else:
            # BERT
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

            self.model.save_pretrained('./objects/')
            self.model = BertForSequenceClassification.from_pretrained('./objects/', from_tf=True)

    def _load_glove(self):
        '''
        Read glove vectors frome the './glove/'
        '''

        f = open('glove/glove.6B.200d.txt','r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

    def get_model(self):
        return self.model

    def load_model(self):

        if os.path.exists('serialization_folder/neuronska.json'):
            self.model = self.load_trained_ann()

    def train_model(self, list_of_pairs, pair_y, X_train_orig, X_train_para):

        left_input = Input(shape=(15,), dtype="int32")
        right_input = Input(shape=(15,), dtype="int32")

        embedding_layer = Embedding(len(self.embeddings), output_dim=self.embedding_dim, weights=[self.embeddings],
                                    input_length=15, trainable=False)

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Define LSTM layer

        shared_lstm = LSTM(128)

        # Propusti enkodovane recenice

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculate distance from the outputs

        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        malstm_model = Model([left_input, right_input], [malstm_distance])


        optimizer = Adadelta(clipnorm = 1.25, lr=0.001)

        def custom_loss(layer):
            def loss(label, distance):
                '''
                label: - 0: not similar
                       - 1: similar
                '''
                label = K.cast(label, tf.float32)
                distance = K.cast(distance, tf.float32)


                # return Tensor(label * 0.5 * (distance ** 2) + (1 - label) * 0.5 (max(0, 1.25 - distance) ** 2))
                return label * 0.5 * K.square(distance) + (1 - label) * 0.5 * K.square(
                    K.maximum(K.zeros_like(distance, dtype=tf.float32), (0.5 - distance)))  # NOT THE EXACT

            return loss

        malstm_model.compile(optimizer=optimizer, loss=custom_loss(malstm_distance))

        malstm_model.summary()

        # Start training
        # TODO
        # jos treba istrenirati mrezu, podaci su spremni, padding odradjen, word embedding isto, sve je stavljeno
        # u liste odgovarajuce duzine
        X_train_orig = [x[0] for x in X_train_orig]
        X_train_para = [x[0] for x in X_train_para]
        X_train_orig = np.array(X_train_orig)
        print(X_train_orig.shape)
        print(X_train_orig[0])
        X_train_para = np.array(X_train_para)
        print(X_train_para.shape)
        pair_y = np.array(pair_y)

        malstm_trained = malstm_model.fit([X_train_orig[:130], X_train_para[:130]], pair_y[:130], batch_size=32, epochs=2000,
        validation_data=([X_train_orig[130:], X_train_para[130:]], pair_y[130:]))


        self.serialize_ann(malstm_model)

        self.model = malstm_model

    def process_input(self, user_input, high_recall_questions):
        '''
        Args:
            user_input (List[String]): the raw user input string
            high_recall_questions ({key, [question, answer]}): the top n results of a high recall model
        Returns:
            dict {key, (distance, [question, answer])}: Questions most similar to the user input
        '''

        if not self.bert:
            preprocessed_input = self._preprocess_user_input(user_input)
            preprocessed_input = [[item] for item in preprocessed_input]

            ret_dict = {}
            for key in high_recall_questions:
                question = self._preprocess_user_input(high_recall_questions[key][0])
                question = [[item] for item in question]

                # WOW, trebalo je konvertovati u tf.Tensor
                dist = self.model([tf.convert_to_tensor(preprocessed_input), tf.convert_to_tensor(question)])
                # print(dist)
                ret_dict[key] = (dist, [high_recall_questions[key]], key)

            # Sort by distance
            ret_dict = {k: v for k, v in sorted(ret_dict.items(), key=lambda item: item[1][0])}

            return ret_dict
        else:
            # BERT pretrained model 
            for key in high_recall_questions:
                question = high_recall_questions[key][0]

                print(f"Processing: {question}")

                model_input = self.tokenizer.encode_plus(user_input, question, add_special_tokens=True, return_tensors='pt')

                prediction = self.model(model_input['input_ids'], token_type_ids=model_input['token_type_ids'])[0]

                print(prediction)
                print(prediction[0].argmax())
                print(prediction[0].argmax().item())
            

    def _preprocess_user_input(self, user_input):
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(user_input,
                                                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                                   lower=True, split=' ')

        # Convert words to their IDs
        word_id_list = []
        for tok in tokens:
            if tok not in self.inv_token_dict.keys():
                continue
            try:
                word_id_list.append(self.inv_token_dict[tok])
            except TypeError:
                pass
            except KeyError:
                word_id_list.append[0]

        # Pad the sequence
        padded = pad_sequences([word_id_list], padding='pre', truncating='post', maxlen=15)
        # print(padded)

        return padded[0]

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

    # print(word_embedding_matrix)

    # word_embedding_matrix = np.asarray(word_embedding_matrix, dtype=np.float32)
    word_embedding_matrix = ([word_embedding_matrix])
    ret_val = pad_sequences(word_embedding_matrix, padding='pre', truncating='post', maxlen=max_seq_len)

    return ret_val


# slicnost izmedju dva izlaza iz sijamske mreze
def exponent_neg_manhattan_distance(left, right):
    # TODO: Zasto eksponent ne valja. Ako je negde 0, onda ce to biti
    # e^0 sto je 1, pa to onda utice na "udaljavanje" 2 vektora (poveca distancu
    # izmedju 2 vektora bespotrebno)
    # print(left.shape)
    # print("Shape is: " + str(left.shape))

    # Shape za train: (160, 15)
    # Shape of user input: (15, 128)
    # NOTE: Verovatno treba malo korigovati process_input
    if left.shape[0] is 15:
        dist = K.sum(K.abs(left - right))
    else:
        dist = K.sum(K.abs(left - right), axis=1)
    return dist