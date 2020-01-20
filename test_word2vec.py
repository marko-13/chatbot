
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import os
import time

from main import index_dataset, save_object, load_object
from preprocessing import preprocess_dataset


def train_w2v(dict):

    dict = index_dataset()

    if os.path.exists('./objects/corpus_token_list.pickle'):
        token_list = load_object('./objects/corpus_token_list.pickle')
    else:
        # Preprocess dataset if needed
        dataset, corpus = preprocess_dataset(dict, lemmatize=True, remove_stopwords=True, measure_time=True)

        token_list = []
        for dataset_entry in dataset:
            token_list.append(list(dataset[dataset_entry][0].keys()))

        save_object(token_list, './objects/corpus_token_list.pickle')

    print(token_list[:5])

    # start = time.time()
    model = Word2Vec(token_list, size=100, window=3, min_count=1, workers=4)
    # print(time.time() - start)

    # print(model.most_similar('plan', topn=10))

    # print(model.wv['insurance'])

    return model
