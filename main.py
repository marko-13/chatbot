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
# koristiti biblioteke
# https://scikit-learn.org/stable/modules/neighbors.html


def save_object(object, filename):
    with open(filename, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


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
    


def main():

    # Read dataset
    dict = index_dataset()

    # Preprocess dataset if needed
    if not os.path.exists('./objects/indexer.pickle') or not os.path.exists('./objects/knn.pickle'):
        dataset, corpus = preprocess_dataset(dict, lemmatize=True, remove_stopwords=True, measure_time=True)

    # Load or create indexer
    if os.path.exists('./objects/indexer.pickle'):
        indexer = load_object('./objects/indexer.pickle')
    else:
        indexer = Indexer(dataset, measure_time=True)
        save_object(indexer, './objects/indexer.pickle')

    #Load or create KNN
    if os.path.exists('./objects/knn.pickle'):
        knn = load_object('./objects/knn.pickle')
    else:
        # Initialize KNN with given dataset
        knn = KNN(dataset, corpus, measure_time=True)
        save_object(knn, './objects/knn.pickle')

    # Main loop for user input
    print("Type a question:")
    q = input()
    while q != 'quit':


        processed_input = preprocess_input(q, lemmatize=True, remove_stopwords=True)

        terms_to_search_for = list(processed_input.keys())

        print('Terms to search for:')
        print(terms_to_search_for)
        print()

        containing_docs = indexer.retrieve_documents(terms_to_search_for, measure_time=True)

        res = knn.find_nearest_neigbours(processed_input, containing_docs , k=10, measure_time=True)

        print("\nResults:\n")
        i = 1
        for r in res:
            print(f'#{i}')
            print(r)
            print()
            i += 1

        print("Type a question:")
        q = input()




if __name__ == "__main__":
    # main()

    dict = index_dataset()
    # dataset, corpus = preprocess_dataset(dict, lemmatize=True, remove_stopwords=True, measure_time=True)
    # print((corpus))
    # print(dict)
    # testing(dict)



    # Load or create indexer
    if os.path.exists('./objects/bot_nn10.pickle'):
        bot = load_object('./objects/bot_nn10.pickle')
    else:
        bot = QnABot()
        bot.set_dataset(dict)
        save_object(bot, './objects/bot_nn10.pickle')

    print("Unesi pitanje:\n")
    q = input()
    while q != 'q':
        bot.process_input(q)
        # print(bot.process_input(q))
        q = input()
