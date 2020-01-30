import tensorflow as tf
from tensorflow import keras

class Dataset():

    def __init__(self, complete_dataset, paraphrazed_dataset, paraphrazed_step=1):
        '''
        The dataset needs to create pairs (original qestion, paraphrazed question),
        labeled with 0s and 1s, which represent if the pair contains similar 
        questions or not.

        Similar questions will have the same key (with regards to the step taken during
        paraphrazing). Example (step = 5):

        paraphrazed.key * step = original.key


        :param dict complete_dataset: The dictionary containing all of the Q/A pairs {key, [qestion, answer]}
        :param dict parapthazed_dataset: {key, paraphrazed_question}
        :param int paraphrazed_step: Every n-th question from the dataset was paraphrazed
        '''
        
        self.complete_dataset = complete_dataset
        self.paraphrazed_dataset = paraphrazed_dataset
        self.paraphrazed_step = paraphrazed_step


        self.similar_pairs = []
        self.non_similar_pairs = []
        
        # Create similar pairs ((paraphrazed_id, q_from_orig_dataset), 0 or 1) 

        for par_q in paraphrazed_dataset:
            orig_key = par_q * self.paraphrazed_step
            pair = ((par_q, orig_key), 1)
            self.similar_pairs.append(pair)




