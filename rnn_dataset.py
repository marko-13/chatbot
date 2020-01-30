import random
from random import randrange

class Dataset():

    def __init__(self, complete_dataset, paraphrazed_dataset, paraphrazed_step=1):
        '''
        NOTE: In this class, we're treating the keys of the complete_dataset as 
              Integers, but they are stored as Strings in the complete_dataset
              dict. Watch out for that!

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
            orig_key = int(par_q) * self.paraphrazed_step
            pair = ((int(par_q), str(orig_key)), 1)
            self.similar_pairs.append(pair)

        # Create non similar pairs

        all_keys = complete_dataset.keys()
        similar_keys = [key for ((par, key), sim) in self.similar_pairs]

        non_similar_keys = [x for x in all_keys if x not in similar_keys]

        for par_q in paraphrazed_dataset:
            random_index = randrange(len(non_similar_keys))
            random_question = non_similar_keys[random_index]
            pair = ((par_q, str(random_question)), 0)

            self.non_similar_pairs.append(pair)

        print("Similar questions: ")
        print(self.similar_pairs[:5])

        print("Non similar questions: ")
        print(self.non_similar_pairs[:5])

        self.mixed_pairs = self.similar_pairs.copy()
        self.mixed_pairs = self.mixed_pairs + self.non_similar_pairs.copy()

        # print(len(self.mixed_pairs))
        # print(self.mixed_pairs[:10])

        self._copy_mixed_pairs()

        print(self.mixed_pairs_copy[:10])


    def get_next_pair(self):
        '''
        Fetch a random datapoint.
        '''
        if self.mixed_pairs_copy == []:
            # Reset the copied mixed pairs list
            self._copy_mixed_pairs() 
            return None

        rand_index = randrange(len(self.mixed_pairs_copy))
        
        return self.mixed_pairs_copy.pop(rand_index)

    def _copy_mixed_pairs(self):
        self.mixed_pairs_copy = self.mixed_pairs.copy()
        random.shuffle(self.mixed_pairs_copy)