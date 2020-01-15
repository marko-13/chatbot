# from sklearn import neigbours

# from scipy import spatial
from numpy import dot
from numpy.linalg import norm

class KNN():

    def __init__(self, dataset, corpus_histogram):

        self.dataset = dataset
        self.corpus_histogram = corpus_histogram
        self.vectorized_dataset = {}

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
                



    def find_nearest_neigbours(self, input, k=10):
        '''

        :param dict input: token frequency dictionary

        '''

        vector = self.vectorize(input)

        results = {}

        # Compute the cos distance between the input 
        # and every question available in the corpus
        for key in self.vectorized_dataset:
            dist = self.cosine(vector, self.vectorized_dataset[key])
            results[key] = dist

        # Sort results by value, descending:
        results_sorted = {k: v for k, v in sorted(results.items(), key = lambda item: item[1], reverse=True)}

        # Return the best k question-answer pairs

        retval = []
        i = 0
        for res in results_sorted:
            print(results_sorted[res])
            retval.append(self.dataset[res])
            i += 1
            if i == k:
                break

        return retval

