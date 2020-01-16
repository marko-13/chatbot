
from numpy import dot
from numpy.linalg import norm

import time

class KNN():

    def __init__(self, dataset, corpus_histogram, measure_time=False):

        self.dataset = dataset
        self.corpus_histogram = corpus_histogram
        self.vectorized_dataset = {}

        if measure_time:
            start = time.time()

        # Vectorize the dataset
        for key in dataset.keys():
            q_token_frequencies, answer = dataset[key]
            question_vector = []
            for word in corpus_histogram:
                if word in q_token_frequencies.keys():
                    # Append the term frequency to the vector
                    question_vector.append(q_token_frequencies[word])
                else:
                    question_vector.append(0)
            self.vectorized_dataset[key] = question_vector

        if measure_time:
            end = time.time()
            print(f'[KNN Initialiation] Elapsed time: {end - start}s')

        sample_vector = next(iter(self.vectorized_dataset.values()))
        print(f"[KNN] Vector size: {len(sample_vector)}")

    def cosine(self, v1, v2):
        '''
        
        :param list v1: vector represented as list
        :param list v2: vector represented as list
        '''
        # return spatial.distance.cosine(v1, v2)
        return dot(v1, v2) / (norm(v1) * norm(v2))

    
    def vectorize(self, token_frequencies):
        '''
        Vectorizes input tokens based on the corpus.
        
        :param dict q_token_frequencies: 
                        key - token (a word)
                        value - it's frequency

        :return  a vector represented as a list
        '''
        retval = []
        for word in self.corpus_histogram:
            if word in token_frequencies:
                retval.append(token_frequencies[word])
            else:
                retval.append(0)
        
        return retval
                



    def find_nearest_neigbours(self, input, document_ids, k=10, measure_time=False):
        '''
        Takes in an input string. Vectorizes it and compares it to all
        of the datapoints in the corpus. Returns the k closest neighbours.


        :param dict input: token frequency dictionary
        :param list document_ids: documents to search from

        '''

        if measure_time:
            start = time.time()

        vector = self.vectorize(input)

        results = {}

        if measure_time:
            compute_start = time.time()

        ''' ======== OLD WAY ===========
        # Compute the cos distance between the input 
        # and every question available in the corpus
        for key in self.vectorized_dataset:
            dist = self.cosine(vector, self.vectorized_dataset[key])
            if dist >= 0.5:
                results[key] = dist
        '''

        for doc in document_ids:
            dist = self.cosine(vector, self.vectorized_dataset[doc])
            if dist >= 0.5:
                results[doc] = dist

        if measure_time:
            print('[KNN] Computation time: {:3.2f}s'.format((time.time() - compute_start)))
            sort_start = time.time()

        # Sort results by value, descending:
        results_sorted = {k: v for k, v in sorted(results.items(), key = lambda item: item[1], reverse=True)}

        if measure_time:
            print('[KNN] Sort time: {:3.2f}'.format((time.time() - sort_start)))

        # Return the best k question-answer pairs

        retval = []
        i = 0
        for res in results_sorted:
            retval.append(self.dataset[res])
            i += 1
            if i == k:
                break

        if measure_time:
            end = time.time()

            elapsed_time = end - start

            print('[KNN] Execution time [k = {}]: {:3.2f}s'.format(k, elapsed_time))

        return retval
