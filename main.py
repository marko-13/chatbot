#from ime_filea import ime_funkcije
import csv
import preprocessing

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

                dict.update({row[0] : pom})
        print(dict)
    return dict



def index_dataset():
    import csv
    
    dict = {}
    

def main():
    print('RUN')
    txt = input("Type something: ")
    print("You wrote: " + txt)



if __name__ == "__main__":
    # main()
    dict = index_dataset()

    dict2 = preprocessing.preprocess_dataset(dict)
    print(dict2)