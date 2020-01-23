#from ime_filea import ime_funkcije
import csv
import os
import pickle


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
    


# def main():

#     # Read dataset
#     dict = index_dataset()

#     # Preprocess dataset if needed
#     if not os.path.exists('./objects/indexer.pickle') or not os.path.exists('./objects/knn.pickle'):
#         dataset, corpus = preprocess_dataset(dict, lemmatize=True, remove_stopwords=True, measure_time=True)

#     # Load or create indexer
#     if os.path.exists('./objects/indexer.pickle'):
#         indexer = load_object('./objects/indexer.pickle')
#     else:
#         indexer = Indexer(dataset, measure_time=True)
#         save_object(indexer, './objects/indexer.pickle')

#     #Load or create KNN
#     if os.path.exists('./objects/knn.pickle'):
#         knn = load_object('./objects/knn.pickle')
#     else:
#         # Initialize KNN with given dataset
#         knn = KNN(dataset, corpus, measure_time=True)
#         save_object(knn, './objects/knn.pickle')

#     # Main loop for user input
#     print("Type a question:")
#     q = input()
#     while q != 'quit':


#         processed_input = preprocess_input(q, lemmatize=True, remove_stopwords=True)

#         terms_to_search_for = list(processed_input.keys())

#         print('Terms to search for:')
#         print(terms_to_search_for)
#         print()

#         containing_docs = indexer.retrieve_documents(terms_to_search_for, measure_time=True)

#         res = knn.find_nearest_neigbours(processed_input, containing_docs , k=10, measure_time=True)

#         print("\nResults:\n")
#         i = 1
#         for r in res:
#             print(f'#{i}')
#             print(r)
#             print()
#             i += 1

#         print("Type a question:")
#         q = input()


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
        ret_ft = bot_ft.process_input(question)

        ret_w2v_ids = [id for id, qa_pair in ret_w2v]
        ret_d2v_ids = [id for id, qa_pair in ret_d2v]
        ret_ft_ids = [id for id, qa_pair in ret_ft]

        results 






if __name__ == "__main__":
    # main()

    dict = index_dataset()
    # print((corpus))
    # print(dict)
    # testing(dict)



    # Load or create indexer
    if os.path.exists('./objects/bot_nn10.pickle'):
        bot = load_object('./objects/bot_nn10.pickle')
    else:
        bot = QnABot()

        # use corpus to find typos in questions
        dataset, corpus = preprocess_dataset(dict, lemmatize=False, remove_stopwords=False, measure_time=True)

        bot.set_dataset(dict, dataset, corpus, algorithm='fasttext')
        save_object(bot, './objects/bot_nn10.pickle')


    q = ""
    while q != 'q':
        q = input("Your question (to quit enter q): ")


        # TODO
        # tokemnizuj input q i za svaku rec nadji najblizu

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
