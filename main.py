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

        bot.set_dataset(dict, corpus)
        save_object(bot, './objects/bot_nn10.pickle')

    q = ""
    while q != 'q':
        q = input("Your question(to quit enter q): ")


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

        ids, ans, question, flag = bot.process_input(q)
        if flag:
            print(f"{ids}, {question} - {ans}")
        else:
            print(f"No suitable answer found")
        # print(bot.process_input(q))
