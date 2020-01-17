from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import NearestNeighbors

from time import time


class QnABot():

    def set_dataset(self, dataset, corpus):
        '''

        :param dict dataset: {key, [question, answer]}
        '''

        self.dataset = dataset

        start = time()

        list_of_q = self._extract_questions(dataset)

        self.vectorizer = CountVectorizer(lowercase=True, analyzer='word')
        X = self.vectorizer.fit_transform(list_of_q)

        self.tf_idf_transformer = TfidfTransformer(use_idf=True).fit(X)

        x_tf_idf = self.tf_idf_transformer.transform(X)

        self.nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', metric='manhattan').fit(x_tf_idf)

        print(start - time())

        self.corpus = corpus
    
    def process_input(self, Y):
        '''

        :param string y: Input STRING
        '''

        # start = time()

        Y = self.vectorizer.transform([Y])

        y_tf_idf = self.tf_idf_transformer.transform(Y)

        distances, ids = self.nbrs.kneighbors(y_tf_idf.todense()) 

        # print(start - time())

        answers = []
        
        for id in ids[0]:
            print(f'{self.dataset[id][0]}')
            print(f'{id} - {self.dataset[id][1]}\n\n')
            
            answers.append(self.dataset[id][1])

        ok_score = False
        for dist in distances[0]:
            if(dist) <= 2:
                ok_score = True
                # return question id, answer, question, and flag
                return ids[0][0], answers[0], self.dataset[ids[0][0]][0], ok_score


        return ids[0][0], answers[0], self.dataset[id][0], ok_score
        

    def _extract_questions(self, dataset):
        ret = []
        for key in dataset:
            val = dataset[key]
            ret.append(str(val[0]))

        return ret

