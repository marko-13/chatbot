from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import NearestNeighbors

from time import time

from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Local imports

from preprocessing import preprocess_input, preprocess_dataset



class QnABot():

    # TODO:
    # Should preprocessing of the dataset questions only involve?
    #   1. tokenization
    #   2. stopword removal
    #   3. conversion to lowercase
    # 
    def set_dataset(self, dataset, processed_dataset, corpus, algorithm='word2vec'):
        '''
        Extracts the questions from the dataset. Runs preprocessing on them
        (which includes tokenization and convesion to lowercase).

        Initializes the chosen algorithm:
            - word2vec (default)
            - doc2vec 
            - fast text

        :param dict dataset: {key, [question, answer]}

        :param string algorithm: The algorithm to use. Possible values: word2vec, doc2vec
        '''

        self.algorithm = algorithm

        self.dataset = dataset

        start = time()

        list_of_q = self._extract_questions(dataset)
        list_of_a = self._extract_answers(dataset)

        
        self.vectorizer = CountVectorizer(lowercase=True, analyzer='word')
        # KNN ON QUESTIONS
        X = self.vectorizer.fit_transform(list_of_q.values())

        # KNN ON ANSWERS
        # NOTE: When training doc2vec on anwsers, it works well only
        # if all of the search terms of the users question are located
        # in the answer. If the answer doesn't include the searched terms,
        # it won't be returned. 
        # X = self.vectorizer.fit_transform(list_of_a.values())
        
        self.tf_idf_transformer = TfidfTransformer(use_idf=True).fit(X)

        x_tf_idf = self.tf_idf_transformer.transform(X)

        self.nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', metric='manhattan').fit(x_tf_idf)
        


        print(f"[Bot] Initialiation of tf-idf and KNN: {time() - start}")

        self.corpus = corpus

        # Extract token list per question
        # for the w2v model
        token_dict = {}
        # TODO: change back to 'list_of_q'
        for key in list_of_q:
            # token_list.append(preprocess_input(question))
            question = list_of_q[key]
            tokens = preprocess_input(question)
            token_dict[key] = tokens


        # === Initialize the model ===

        # Train w2v model
        start = time()
        if algorithm == 'word2vec':
            q_tokens_list = list(token_dict.values())
            self.model = Word2Vec(q_tokens_list, size=100, window=3, min_count=0, workers=4, iter=10)
        elif algorithm == 'doc2vec':
            documents = [TaggedDocument(doc, [key]) for key, doc in token_dict.items()]
            self.model = Doc2Vec(documents,  vector_size=150, window=3, min_count=0, workers=4, epochs=10)
        elif algorithm == 'fasttext':
            # Word2Vec sa n-gramima
            q_tokens_list = list(token_dict.values())
            self.model = FastText(size=100, window=3, min_count=0, sentences=q_tokens_list, iter=10)
        
        print(f"[Bot] Model training time: {time() - start}")

    
    def process_input(self, Y):
        '''

        :param string y: Input STRING
        '''

        start = time()

        # Save the raw input string
        raw_input = Y

        #  === High Recall model ===
        # Get possibly relevant documents using 
        # tf-idf vectors and the KNN algorithm

        Y = self.vectorizer.transform([Y])

        y_tf_idf = self.tf_idf_transformer.transform(Y)

        distances, ids = self.nbrs.kneighbors(y_tf_idf.todense()) 
        # typeof(ids) is [[Int]]

        k_nearest_ids = ids[0]

        # ============================

        # Compute similarity scores for all k nearest questions
        top_hits = self._compute_similarity(raw_input, k_nearest_ids)

        return top_hits


    def _compute_similarity(self, raw_input, ids, n_hits=10):
        '''
        Computes the similarity scores between the user input and 
        a collection of datapoints in the dataset.

        The similarity is computed as follows:

            - for each question in the collection
                - for each word in the input
                    - the max similarity is found between the input word
                      and all the words in the current question
                    - the max similarity is summed up for the current question
        
        Now all of the datapoints have an associated similarity score. 
        The datapoints are sorted by that score and the function returns
        the top n hits.
        

        :param string input: The raw input string
        :param list ids: The IDs of the documents to which to compare the input.
        :param int n_hits: Number of top hits to return
        :return list top_hits: Returns pairs (question, answer) who have the most similar
                               questions to the input
        '''
        # Rate questions by their similarity scores using w2v
        q_similarity_scores = {}
        input_tokens = preprocess_input(raw_input)
        if self.algorithm == 'word2vec':
            for id in ids:
                question = self.dataset[id][0]
                sum_similarities = 0
                
                question = preprocess_input(question)

                
                for word in input_tokens:

                    for q_word in question:
                        try:
                            # Find the maximum similarity of each input word
                            # to each word in the question
                            sims = [self.model.wv.similarity(word, q_word)]
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
        elif self.algorithm == 'doc2vec':
            input_tokens = list(input_tokens.keys())
            # print(input_tokens)
            doc_vec = self.model.infer_vector(input_tokens, alpha=0.01, epochs=100)
            # print(self.model.docvecs[13])

            sims = self.model.docvecs.most_similar([doc_vec])

            # print(sims)

            # Return the top 10 results
            retval = []
            i = 0
            for id, cos_sim in sims:
                retval.append((id, self.dataset[id]))
                i += 1
                if i == 10:
                    break

            return retval
        elif self.algorithm == 'fasttext':
            input_tokens = list(input_tokens.keys())

            sims = self.model.most_similar(positive=input_tokens, topn=10)

            print(sims)

            # Return the top 10 results
            retval = []
            i = 0
            for id, cos_sim in sims:
                retval.append((id, self.dataset[id]))
                i += 1
                if i == 10:
                    break

            return retval


    def _extract_questions(self, dataset):
        '''
        Extracts only the questions from the dataset

        :return dict: {q_id, question}
        '''
        ret = {}
        for key in dataset:
            val = dataset[key]
            # ret.append(str(val[0]))
            ret[key] = str(val[0])

        return ret


    def _extract_answers(self, dataset):
        '''
        Extracts only the questions from the dataset

        :return dict: {q_id, question}
        '''
        ret = {}
        for key in dataset:
            val = dataset[key]
            # ret.append(str(val[0]))
            ret[key] = str(val[1])

        return ret