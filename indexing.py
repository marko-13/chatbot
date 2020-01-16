import time

class Indexer():

    def __init__(self, dataset, measure_time=False):
        '''
        Indexes the corpus via inverted index.
        
        :param dict dataset - {documentId, [term_frequencies, answer]}
                              term_frequencies: {term, frequency}
        '''

        # Key-value
        # Term-List of Document IDs
        self.inverted_index = {}

        if measure_time:
            start = time.time()

        # Create the inverted index
        for documentId in dataset.keys():
            terms = dataset[documentId][0].keys()
            print(f"[Indexer] Terms in document {documentId}")
            print(dataset[documentId][0])
            for term in terms:
                if term in self.inverted_index:
                    containing_docs = self.inverted_index[term]
                    self.inverted_index[term] = list(set(containing_docs).union([documentId]))
                else:
                    self.inverted_index[term] = [documentId]

        print(f"[Indexer] Number of terms: {len(list(self.inverted_index.keys()))}")

        if measure_time:
            end = time.time()
            print('[Indexer] Indexing time: {:3.2f}'.format((end - start)))

    
    def retrieve_documents(self, terms, num_docs=1000, measure_time=False):
        '''
        Retrieve all the documents containing ALL OF the supplied terms.

        :param list terms: List of terms to search for
        '''

        print(f'Number of indexed terms: {len(list(self.inverted_index.keys()))}')

        retval = []

        if measure_time:
            start = time.time()

        for term in terms:
            if term in self.inverted_index:
                containing_documents = self.inverted_index[term]
                print(f'Num documents containig {term}:')
                print(len(containing_documents))
                print()

                # TODO: intersect lists?
                if retval == []:
                    retval = containing_documents
                else:
                    retval = list(set(retval) & set(containing_documents))


        if measure_time:
            end = time.time()
            print('[Indexer] Retrieval time: {:3.2f}'.format((end - start)))


        return retval
