import csv
import os
import sys
import pickle
import datetime

from matplotlib import pyplot as plt
from nearest_neighbours import KNN

# Local imports
from preprocessing import preprocess_dataset, preprocess_input
from bot import QnABot
from indexing import Indexer
from test import testing

from rnn_dataset import Dataset
# ----------------------------------------------------------------------------------------------------------------------
# TODO
# Add weights to check similarity in word2vec similarity in typed question and list of questions
# Note: will be using pretrained GloVe model, trained on wikipedia 800mb in size


def save_object(object_to_save, filename):
    with open(filename, 'wb') as output:
        pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


def index_dataset():
    dict = {}

    with open('insurance_qna_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            # Skip first line
            if line_count == 0:
                line_count = line_count + 1
            else:
                # If answer(row[2]) does not exist, skip
                if row[2] == '':
                    continue
                pom = []
                pom.append(row[1])
                pom.append(row[2].rstrip())
                dict.update({int(row[0]): pom})
    return dict


def read_paraphrazed_dataset(filename):
    dict = {}

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            dict[int(row[0])] = row[1]
    return dict


def main(dict, algorithm_v1):
    # Get desired algorithm bot
    bot = get_bot(dict, algorithm=algorithm_v1)

    q = input("Your question (to quit enter q): ")
    while q != 'q':

        for ret in bot.process_input(q):
            print(f"{ret[0]})")
            print(ret[1])
            print()
            print()

        q = input("Your question (to quit enter q): ")


def get_bot(dataset_dict, algorithm='word2vec', do_lemmatization=True):
    # Load bot model if possible, create new otherwise
    if os.path.exists(f'./objects/bot_{algorithm}.pickle'):
        bot = load_object(f'./objects/bot_{algorithm}.pickle')
    else:
        bot = QnABot()
        dataset, corpus = preprocess_dataset(dataset_dict, lemmatize=do_lemmatization, remove_stopwords=False,
                                             measure_time=True)

        bot.set_dataset(dataset_dict, dataset, corpus, algorithm=algorithm, lemmatize=do_lemmatization)
        save_object(bot, f'./objects/bot_{algorithm}.pickle')

    return bot


def run_comparison_testing(dataset_dict, wanted_questions):
    '''
    :param dict dataset_dict: indexed dataset dictionary
    :param list wanted_questions: list of tuples (question, [ids])
        which represent our users input and the id of the 
        desired question/answer pair we want our bot to return
    '''
    # dataset_dict = {id, [question, answer]}

    lemmatize = True

    bot_w2v = get_bot(dataset_dict, 'word2vec', lemmatize)
    bot_d2v = get_bot(dataset_dict, 'doc2vec', lemmatize)
    bot_ft = get_bot(dataset_dict, 'fasttext', lemmatize)
    bot_w2v_sum = get_bot(dataset_dict, 'word2vecSum', lemmatize)
    bot_ft_sum = get_bot(dataset_dict, 'fasttextSum', lemmatize)

    # A list of pairs our desired output's ID and it's 
    # location in the top 10 of our 3 algorithms
    # [(id, [pos_w2v, pos_d2v, pos_ft])]
    results = []
    for question, id_list in wanted_questions:
        # list of (id, [question, answer])
        ret_w2v = bot_w2v.process_input(question)
        ret_d2v = bot_d2v.process_input(question)
        ret_ft = bot_ft.process_input(question)
        ret_w2v_sum = bot_w2v_sum.process_input(question)
        ret_ft_sum = bot_ft_sum.process_input(question)

        ret_w2v_ids = [id for id, qa_pair in ret_w2v]
        ret_d2v_ids = [id for id, qa_pair in ret_d2v]
        ret_ft_ids = [id for id, qa_pair in ret_ft]
        ret_w2v_sum_ids = [id for id, qa_pair in ret_w2v_sum]
        ret_ft_sum_ids = [id for id, qa_pair in ret_ft_sum]


        all_ret = [
            ret_w2v_ids,
            ret_d2v_ids,
            ret_ft_ids,
            ret_w2v_sum_ids,
            ret_ft_sum_ids
        ]

        positions = []
        
        for alg in all_ret:
            indices = []
            found = False
            for alg_retval in alg:
                if alg_retval in id_list:
                    positions.append(alg.index(alg_retval))
                    found = True
                    break
            if not found:
                positions.append(-1)

        print(question)
        print(positions)
        print("\n")

        print(id_list[0])

        print("Word2Vec output:")
        for id in ret_w2v_ids:
            print(id)
            print(dataset_dict[id][0])
            print('\n')
        print('==============\n')

        print("Doc2Vec output:")
        for id in ret_d2v_ids:
            print(id)
            print(dataset_dict[id][0])
            print('\n')
        print('==============\n')

        print("FastText output:")
        for id in ret_ft_ids:
            print(id)
            print(dataset_dict[id][0])
            print('\n')
        print('==============\n')
        
        results.append((id_list[0], positions))

    print(results)


    # PLOTTING
    x_ax = []
    y_ax = []
    for id, pos in results:
        y_ax.append(pos)

    labels = [str(q[1][0]) for q in wanted_questions]
    
    x_range = list(range(len(results)))

    horizontal_line = [0 for i in x_range]

    plt.yticks(list(range(-1, 10)))
    plt.xticks(x_range, labels)


    plt.plot(x_range, horizontal_line)
    plt.plot(x_range, [val[0] for val in y_ax], "s", color='b', label="Word2Vec", markersize=12)
    plt.plot(x_range, [val[1] for val in y_ax], 'o', label="Doc2Vec", markersize=10)
    plt.plot(x_range, [val[2] for val in y_ax], 'v', label="FastText", markersize=12)
    plt.plot(x_range, [val[3] for val in y_ax], "s", label="Word2VecSum", markersize=6)
    plt.plot(x_range, [val[4] for val in y_ax], 'v', label="FastTextSum", markersize=6)

    plt.legend(loc="upper left")

    plt.savefig('log/with_lemm.png')
    plt.show()




def testing():

    print("Poceo u: " + str(datetime.datetime.now()))

    with open('insurance_qna_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            # Skip first line
            if line_count == 0:
                line_count = line_count + 1
            else:
                # If answer(row[2]) does not exist, skip
                if row[2] == '':
                    continue



    # read csv file and return {id, [question, answer]}
    dict = index_dataset()

    # dict = {id, [question, amnswer]}, desired algorithm name("fasttext", "doc2vec", "word2vec")
    # main(dict, "word2vec")

    # Set algorithm from command line argument
    algorithm = 'word2vec' if len(sys.argv) < 2 else sys.argv[1]
    print("Used algorithm: " + str(algorithm))

    # Questions for testing accuracy
    wanted_questions = [("Is fixed annuity safe?", [15453]), 
                        ("Can I opt out of health insurance", [17884]), 
                        ("How to become a medicare expert", [22917, 22918]), 
                        ("Can disability insurance be deducted", [25381]),
                        ("Is scuba diving covered by life insurance", [num for num in range(20565, 20569)]),
                        ("is there life insurance for elderly", [25379, 25380]),
                        ("blanket life insurance", [9783]),
                        ("What is jumbo life insurance", [num for num in range(25650, 25654)]),
                        ("can i get a life insurance policy from my parent", [11963, 11964, 11965, 11966, 15786, 17280, 17281, 27635, 27636, 27637, 2465, 2466, 2467, 7215, 7216, 7399, 7400]),
                        ("cashing out a 401k", [49, 50, 1708, 5147, 5428, 7788, 7789, 12410]),
                        ("disability insurance and pregnancy", [746, 2516, 2517, 27196, 27197, 26405, 26314, 26315, 26069, 25980, 25981, 20765, 20435, 19122, 19123, 19124, 19125, 18062, 18063, 17447, 17448, 17004, 17005, 17006, 10633, 10634, 1253]),
                        ]# ("", []),
                        # ("", []),
                        # ("", []),
                        # ("", [])]

    run_comparison_testing(dict, wanted_questions)

# ------------------------------------------------------------------------------------------------

def rnn_training():
    # read csv file and return {id, [question, answer]}
    dict_orig = index_dataset()

    dict_paraphrazed = read_paraphrazed_dataset('paraphrazed_final_100.csv')

    dataset = Dataset(dict_orig, dict_paraphrazed, 5)





# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    rnn_training()
    # testing()

