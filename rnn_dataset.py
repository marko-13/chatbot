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

        self.num_instances = len(self.paraphrazed_dataset)

        train_num_inst = int(self.num_instances * 0.8)

        # Define the train set
        self.mixed_pairs_train = self.similar_pairs.copy()[:train_num_inst]
        self.mixed_pairs_train = self.mixed_pairs_train + self.non_similar_pairs.copy()[:train_num_inst]

        # Define the test set
        self.mixed_pairs_test = self.similar_pairs.copy()[train_num_inst:]
        self.mixed_pairs_test = self.mixed_pairs_test + self.non_similar_pairs.copy()[train_num_inst:]

        # print(len(self.mixed_pairs))
        # print(self.mixed_pairs[:10])

        # Copy the training and testing sets
        self._copy_mixed_pairs()
        self._copy_mixed_pairs(train=False)

        print('Training set:')
        print(len(self.mixed_pairs_train_copy))
        print(self.mixed_pairs_train_copy[:10])

        print('Test set:')
        print(len(self.mixed_pairs_test_copy))
        print(self.mixed_pairs_test_copy[:10])


    def get_next_pair(self):
        '''
        Fetch a random datapoint from the training set. Will return None if it is empty, 
        and will reset the list, for use in the next epoch
        '''
        if self.mixed_pairs_train_copy == []:
            # Reset the copied mixed pairs list
            self._copy_mixed_pairs() 
            return None

        rand_index = randrange(len(self.mixed_pairs_train_copy))
        
        return self.mixed_pairs_train_copy.pop(rand_index)

    def get_next_pair_test(self):
        if self.mixed_pairs_test_copy == []:
            # Reset the copied mixed pairs list
            self._copy_mixed_pairs(train=False) 
            return None

        rand_index = randrange(len(self.mixed_pairs_test_copy))
        
        return self.mixed_pairs_test_copy.pop(rand_index)

    def _copy_mixed_pairs(self, train=True):
        if train:
            self.mixed_pairs_train_copy = self.mixed_pairs_train.copy()
            random.shuffle(self.mixed_pairs_train_copy)
        else:
            self.mixed_pairs_test_copy = self.mixed_pairs_test.copy()
            random.shuffle(self.mixed_pairs_test_copy)

    def get_original_question(self, key):
        return self.complete_dataset[key][0]

    def get_paraphrized_question(self, key):
        return self.paraphrazed_dataset[key]

    def get_original_dataset_keys(self):
        return self.complete_dataset.keys()

    def get_original_dataset_questions(self):
        retval = []
        for key in self.get_original_dataset_keys:
            retval.append(self.complete_dataset[key][0])
        return retval
    
