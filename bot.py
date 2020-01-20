from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import NearestNeighbors

from time import time

from gensim.models import Word2Vec

# Local imports

from preprocessing import preprocess_input, preprocess_dataset



class QnABot():

    def set_dataset(self, dataset, processed_dataset, corpus):
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

        # Extract token list per question
        # for the w2v model
        token_list = []
        for question in list_of_q:
            token_list.append(preprocess_input(question))

        # Train w2v model
        self.w2v_model = Word2Vec(token_list, size=100, window=3, min_count=0, workers=4, iter=10)

    
    def process_input(self, Y):
        '''

        :param string y: Input STRING
        '''

        start = time()

        # Save the raw input string
        raw_input = Y

        Y = self.vectorizer.transform([Y])

        y_tf_idf = self.tf_idf_transformer.transform(Y)

        distances, ids = self.nbrs.kneighbors(y_tf_idf.todense()) 

        # Rate questions by their similarity scores using w2v
        q_similarity_scores = {}
        input_tokens = preprocess_input(raw_input)
        for id in ids[0]:
            question = self.dataset[id][0]
            sum_similarities = 0
            
            question = preprocess_input(question)
            
            for word in input_tokens:

                for q_word in question:
                    try:
                        # Find the maximum similarity of each input word
                        # to each word in the question
                        sims = [self.w2v_model.wv.similarity(word, q_word)]
                        sum_similarities += max(sims)
                    except KeyError:
                        print(f"Word {q_word} not in dataset")
            
            # Associate every question with it's similarity
            # to the input
            q_similarity_scores[id] = sum_similarities

        # Sort question IDs by their similarity to the input
        sorted_by_sim = {id: sim for id, sim in sorted(q_similarity_scores.items(), key = lambda x: x[1], reverse=True) }

        # Return the top 10 results
        retval = []
        i = 0
        for res in sorted_by_sim:
            retval.append((res, self.dataset[res]))
            i += 1
            if i == 10:
                break

        return retval


    def _extract_questions(self, dataset):
        ret = []
        for key in dataset:
            val = dataset[key]
            ret.append(str(val[0]))

        return ret

