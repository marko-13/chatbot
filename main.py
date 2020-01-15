#from ime_filea import ime_funkcije
import csv

# Local imports
from preprocessing import preprocess_dataset, preprocess_input

from nearest_neighbours import KNN

def index_dataset():
    dict = {}

    with open('dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
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
                pom.append(row[2])

                dict.update({int(row[0]) : pom})
        # print(dict)
    return dict
    

def main():
    print('RUN')
    txt = input("Type something: ")
    print("You wrote: " + txt)



if __name__ == "__main__":
    # main()
    dict = index_dataset()



    # print(dict)
    # print(dict[1])

    # Initialize dataset
    dict2, corpus = preprocess_dataset(dict, lemmatize=True, remove_stopwords=True, measure_time=True)

    # print(dict2[1])

    knn = KNN(dict2, corpus)

    print("Type a question:")
    q = input()
    while q != 'quit':
        print("Result:")

        res = knn.find_nearest_neigbours(preprocess_input(q), measure_time=True)

        for r in res:
            print(r)

        print("Type a question:")
        q = input()
