#from ime_filea import ime_funkcije
import csv
import os
import pickle
import datetime
from matplotlib import pyplot as plt

# Local imports
from preprocessing import preprocess_dataset, preprocess_input

from nearest_neighbours import KNN

from indexing import Indexer

from test import testing

from bot import QnABot

# TODO:
# FAST TEXT i DOC2VEC


def save_object(object, filename):
    with open(filename, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def get_bot(dataset_dict, algorithm='word2vec' ):
    
    # Load or create indexer
    if os.path.exists(f'./objects/bot_{algorithm}.pickle'):
        bot = load_object(f'./objects/bot_{algorithm}.pickle')
    else:
        bot = QnABot()

        # use corpus to find typos in questions
        dataset, corpus = preprocess_dataset(dataset_dict, lemmatize=False, remove_stopwords=False, measure_time=True)

        bot.set_dataset(dataset_dict, dataset, corpus, algorithm=algorithm)
        save_object(bot, f'./objects/bot_{algorithm}.pickle')

    return bot


def index_dataset():
    dict = {}

    with open('insurance_qna_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                # print(f'{row[0]}  {row[1]}  {row[2]}')
                # ako nema odgovor prskoci
                if row[2] == '':
                    continue
                pom = []
                pom.append(row[1])
                pom.append(row[2].rstrip())
                dict.update({int(row[0]) : pom})
        # print(dict)
    return dict
    


def main(dict):

    print(datetime.datetime.now())

    # Load or create indexer
    if os.path.exists('./objects/bot_nn10.pickle'):
        bot = load_object('./objects/bot_nn10.pickle')
    else:
        bot = QnABot()

        # use corpus to find typos in questions
        dataset, corpus = preprocess_dataset(dict, lemmatize=False, remove_stopwords=False, measure_time=True)

        bot.set_dataset(dict, dataset, corpus, algorithm='word2vec')
        save_object(bot, './objects/bot_nn10.pickle')


    q = ""
    while q != 'q':
        q = input("Your question (to quit enter q): ")

        # check for typos
        flag_typos = True
        all_incorrect = True
        # TODO
        # proveri da li je samo typo ili je cela recenica neka brljotina, ako je samo typo uradi levenshteina
        split_str = q.split(" ")
        for word in split_str:
            if word.lower() in bot.corpus:
                flag_typos = False

        if flag_typos:
            print("No suitable answers found.\n")
            continue

        # ids, ans, question, flag = bot.process_input(q)
        # if flag:
        #     print(f"{ids}, {question} - {ans}")
        # else:
        #     print(f"No suitable answer found")
        # print(bot.process_input(q))

        for ret in bot.process_input(q):
            print(f"{ret[0]})")
            print(ret[1])
            print()
            print()



def run_comparison_testing(dataset_dict, wanted_questions):
    '''
    
    :param dict dataset_dict: indexed dataset dictionary
    :param list wanted_questions: list of tuples (question, id)
        which represent our users input and the id of the 
        desired question/answer pair we want our bot to return
    '''
    bot_w2v = get_bot(dataset_dict, 'word2vec')
    bot_d2v = get_bot(dataset_dict, 'doc2vec')
    bot_ft = get_bot(dataset_dict, 'fasttext')

    # A list of pairs our desired output's ID and it's 
    # location in the top 10 of our 3 algorithms
    # [(id, [pos_w2v, pos_d2v, pos_ft])]
    results = []
    for question, id in wanted_questions:
        # list of (id, [question, answer])
        ret_w2v = bot_w2v.process_input(question)
        ret_d2v = bot_d2v.process_input(question)
        # ret_ft = bot.process_input(question)

        ret_w2v_ids = [id for id, qa_pair in ret_w2v]
        ret_d2v_ids = [id for id, qa_pair in ret_d2v]
        # ret_ft_ids = [id for id, qa_pair in ret_ft]

        positions = []
        
        for alg in [ret_w2v_ids, ret_d2v_ids]:
            try:
                pos = alg.index(id)
                positions.append(pos)
            except ValueError:
                positions.append(-1)

        print(question)
        print(positions)
        print("\n")

        print("Word2Vec output:")
        for id in ret_w2v_ids:
            print(dataset_dict[id][1])
            print('\n')
        print('==============\n')

        print("Doc2Vec output:")
        for id in ret_d2v_ids:
            print(dataset_dict[id][1])
            print('\n')
        print('==============\n')
        
        results.append((id, positions))

    print(results)

    x_ax = []
    y_ax = []
    for id, pos in results:
        y_ax.append(pos)

    labels = [str(q[1]) for q in wanted_questions]
    
    x_range = list(range(len(results)))

    plt.yticks(list(range(-1, 10)))
    plt.xticks(x_range, labels)


    
    plt.plot(x_range, [val[0] for val in y_ax], 'o', label="Word2Vec")
    plt.plot(x_range, [val[1] for val in y_ax], 'o', label="Doc2Vec")

    plt.legend(loc="upper left")

    plt.show()






if __name__ == "__main__":

    dict = index_dataset()

    # main()

    wanted_questions = [("Is fixed annuity safe?", 15453), \
                        ("Can I opt out of health insurance",17884), \
                        ("How to become a medicare expert", 22918), \
                        ("Can you deduct disability insurance", 25381)]

    run_comparison_testing(dict, wanted_questions)

