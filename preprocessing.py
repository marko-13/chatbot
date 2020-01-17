import nltk
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import time



nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
print('\n\n')


def lemmatize_tokens(tokens):
    '''
    Performs lemmatization on a list of tokens
    '''

    lemmatizer = WordNetLemmatizer()

    lem_tokens = []

    for t in tokens:
        lem_t = lemmatizer.lemmatize(t)
        if (t != lem_t):
            lem_tokens.append(lem_t)
        else:
            lem_tokens.append(t)
    
    return lem_tokens


def tokens_to_lower(tokens):
    '''
    Transforms all tokens in a list to lower case.
    '''
    retval = []
    for t in tokens:
        retval.append(t.lower())
    
    return retval


def token_frequency(tokens):
    '''
    Returns a histogram of token occurences
    in a list.
    '''
    token_freq = {}
    for t in tokens:
        if t in token_freq:
            token_freq[t] += 1
        else:
            token_freq[t] = 1
    
    return token_freq


def preprocess_dataset(dataset, lemmatize=False, remove_stopwords=False, measure_time=False):
    '''
    Takes a dataset dictionary, performs preprocessing on it's questions,
    which involves:
        1. Tokenization
        2. Lemmatization (optional)
        3. Stop word removal (optional)
        4. Converts all tokens to lower case
    
    Additional tasks:
        - Returns a word histogram of the entire corpus

    Returns 2 values:
        1. Processed dataset: {key, [term_frequencies, answer]}
                - term frequencies: a dictionary (histogram)
                                    of all token occurences in the
                                    datapoint
        2. Corpus histogram

    :param dict dataset - {key, [question, answer]}

    :return dict processed_dataset - {key, [term_frequencies, answer]}
    '''
    
    corpus_histogram = {}

    processed_dataset = {}

    if measure_time:
        global_time = 0

    for key in dataset.keys():
        '''
        For every question-answer pair, do preprocessing and update
        the global corpus dictionary
        '''
        if measure_time:
            start = time.time()

        val = dataset[key] 
        question, answer = val[0], val[1]
        
        tokens = nltk.word_tokenize(question)

        # Convert to lower case
        tokens = tokens_to_lower(tokens)

        if lemmatize:
            tokens = lemmatize_tokens(tokens)
        
        if remove_stopwords:
            tokens = [token for token in tokens if not token in stopwords.words('english')]
        
        if measure_time:
            end = time.time()
            global_time += (end - start)


        # Update the global corpus

        for t in tokens:
            if t in corpus_histogram:
                corpus_histogram[t] += 1
            else:
                corpus_histogram[t] = 1

        # Measure token frequency

        token_freq = token_frequency(tokens)

        # Return token frequencies and associated anwsers

        processed_dataset[key] = [token_freq, answer]

    if measure_time:
        print(f'[Dataset Preprocessing] Time elapsed is: {global_time}')

    return processed_dataset, corpus_histogram



def preprocess_input(input_str, lemmatize=False, remove_stopwords=False):
    '''
    Takes an input string and does preprocessing on it:
        1. Tokenization
        2. Lemmatization
        3. Stop word removal
        4. Term frequency count

    Return a dictionary of term frequencies for each term.
    '''
    tokens = nltk.word_tokenize(input_str)

    tokens = tokens_to_lower(tokens)

    if lemmatize:
        lemmatize_tokens(tokens)
    if remove_stopwords:
        tokens = [token for token in tokens if not token in stopwords.words('english')]


    token_freq = token_frequency(tokens)

    return token_freq
