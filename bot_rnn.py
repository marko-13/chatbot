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

import torch

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

        4. (NOVO) Treba koristiti BeroTokenizer.encode()
    '''

    def __init__(self, list_of_pairs, pair_y, train=False, bert=False, hybrid=False):

        self.bert = bert
        self.hybrid = hybrid

        if bert:
            # BERT
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

            self.model.save_pretrained('./objects/')
            self.model = BertForSequenceClassification.from_pretrained('./objects/', from_tf=True)

        elif hybrid:
            # Use the BERT encoder to train our RNN
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer.padding_side = 'left'
            self.bert_model = BertModel.from_pretrained('bert-base-cased')
            max_input_len = 40
  

         
            X_train_para = []
            X_train_orig = []

            # for question in Dataset.get_original_dataset_questions(load_object("objects/rnn_dataset_object.pickle")):
                # print(question)

            def encode_fun(x):
                # Enkodiranje svake reci pomocu BERT-a 
                input_ids = torch.tensor(self.tokenizer.encode(x, max_length=40, pad_to_max_length=True)).unsqueeze(0)
                outputs = self.bert_model(input_ids)
                # Uzimamo poslednji output, pretvaramo u numpy
                last_hidden_states = outputs[0][0].detach().numpy()
                return last_hidden_states



            for pair in list_of_pairs:
                X_train_para.append(pair[0])
                X_train_orig.append(pair[1])
                # X_train_para.append
                # pass
            # encoded_right = self.tokenizer.encode(right_input, max_length=max_input_len, pad_to_max_length=True).unsqueeze(0)


            X_train_orig = [encode_fun(x) for x in X_train_orig]
            X_train_para = [encode_fun(x) for x in X_train_para]
            print(X_train_para[0])
            X_train_orig = np.array(X_train_orig)
            X_train_para = np.array(X_train_para)

            X_train_orig = K.constant(X_train_orig)
            X_train_para = K.constant(X_train_para)

            print(f"Shape: {X_train_orig.shape}")
            print(X_train_para[0])
            if train:
                self.train_model(list_of_pairs, pair_y, X_train_orig, X_train_para)
            else:
                self.load_model('rnn_bert')

        else:
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

    def load_model(self, name):

        if os.path.exists(f'serialization_folder/{name}.json'):
            self.model = self.load_trained_ann(name)

    def train_model(self, list_of_pairs, pair_y, X_train_orig, X_train_para):

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

        if self.hybrid:

            max_input_len = 40


            left_input = Input(shape=(max_input_len,768), dtype="float32")
            right_input = Input(shape=(max_input_len,768), dtype="float32")

            
            
            # Define LSTM layer

            shared_lstm = LSTM(128)

            # Propusti enkodovane recenice

            left_output = shared_lstm(left_input)
            right_output = shared_lstm(right_input)


            malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                    output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

            malstm_model = Model([left_input, right_input], [malstm_distance])


            optimizer = Adadelta(clipnorm = 1.25, lr=0.001)

            

            malstm_model.compile(optimizer=optimizer, loss=custom_loss(malstm_distance))

            malstm_model.summary()

            # X_train_orig = [x[0] for x in X_train_orig]
            # X_train_para = [x[0] for x in X_train_para]
            # print(X_train_orig.shape)
            # print(X_train_orig[0])
            # print(X_train_para.shape)
            pair_y = np.array(pair_y)

            # Embed using bert:

            print(f"Shape 2: {X_train_orig.shape}")

            malstm_trained = malstm_model.fit([X_train_orig[:130], X_train_para[:130]], pair_y[:130], batch_size=32, epochs=2000,
            validation_data=([X_train_orig[130:], X_train_para[130:]], pair_y[130:]))


            self.serialize_ann(malstm_model, 'rnn_bert')

            self.model = malstm_model

        else:


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


            self.serialize_ann(malstm_model, 'rnn_w2v')

            self.model = malstm_model

    def process_input(self, user_input, high_recall_questions):
        '''
        Args:
            user_input (List[String]): the raw user input string
            high_recall_questions ({key, [question, answer]}): the top n results of a high recall model
        Returns:
            dict {key, (distance, [question, answer])}: Questions most similar to the user input
        '''
        if self.hybrid:

            print("Processing input as hybrid network...")
            def encode_fun(x):
                # Enkodiranje svake reci pomocu BERT-a 
                input_ids = torch.tensor(self.tokenizer.encode(x, max_length=40, pad_to_max_length=True)).unsqueeze(0)
                print(f"Encoded question:\n{input_ids}")
                outputs = self.bert_model(input_ids)
                # Uzimamo poslednji output, pretvaramo u numpy
                last_hidden_states = outputs[0][0].detach().numpy()
                return last_hidden_states

            encoded_input = np.array([encode_fun(user_input)])
            print(f"Encoded input shape: {encoded_input.shape}")

            ret_dict = {}
            for key in high_recall_questions:
                question = high_recall_questions[key][0]

                print(f"Processing {question}")
                encoded_question = np.array([encode_fun(question)])
                print(f"Encoded question shape: {encoded_question.shape}")
                model_input = [encoded_input, encoded_question]
                # print(f"Model input shape: {model_input.shape}")
                dist = self.model(model_input)

                # dist = prediction[0][1]
                ret_dict[key] = (dist, [high_recall_questions[key]], key)

            ret_dict = {k: v for k, v in sorted(ret_dict.items(), key=lambda item: item[1][0])}

            return ret_dict

        elif not self.bert:
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
            ret_dict = {}
            for key in high_recall_questions:
                question = high_recall_questions[key][0]

                print(f"Processing: {question}")

                model_input = self.tokenizer.encode_plus(user_input, question, add_special_tokens=True, return_tensors='pt')

                prediction = self.model(model_input['input_ids'], token_type_ids=model_input['token_type_ids'])[0]

                similarity_probability = prediction[0][1]

                
                print(prediction)
                print(prediction[0].argmax())
                print(prediction[0].argmax().item())
                print()
                
                ret_dict[key] = (similarity_probability, [high_recall_questions[key]], key)


            ret_dict = {k: v for k, v in sorted(ret_dict.items(), key=lambda item: item[1][0])}

            return ret_dict
                # print(prediction)
                # print(prediction[0].argmax())
                # print(prediction[0].argmax().item())
            

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

    def serialize_ann(self, rnn, name):
        # serijalizuj arhitekturu neuronske mreze u JSON fajl
        model_json = rnn.to_json()
        with open(f"serialization_folder/{name}.json", "w") as json_file:
            json_file.write(model_json)
        # serijalizuj tezine u HDF5 fajl
        rnn.save_weights(f"serialization_folder/{name}.h5")

    def load_trained_ann(self, name):
        try:
            # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
            json_file = open(f'serialization_folder/{name}.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            ann = model_from_json(loaded_model_json)
            # ucitaj tezine u prethodno kreirani model
            ann.load_weights(f"serialization_folder/{name}.h5")
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
    print(f"Computing Manhattan for input tensors: {left.shape} {right.shape}")
    if left.shape[0] is 15:
        dist = K.sum(K.abs(left - right))
    elif left.shape[0] is 40:
        dist = K.sum(K.abs(left - right))
    else:
        dist = K.sum(K.abs(left - right), axis=1)
    return dist