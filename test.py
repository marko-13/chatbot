from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def extract_questions(corpus):
    ret = []
    for key in corpus:
        val = corpus[key]
        # question, answer = val[0], val[1]
        # print(val[0])
        ret.append(str(val[0]))

    return ret

def testing(corpus):

    list_of_q = extract_questions(corpus)
    print(list_of_q)

    vectorizer = CountVectorizer(lowercase=True, analyzer='word')
    X = vectorizer.fit_transform(list_of_q)
    print(X.shape)
    # print(len(vectorizer.get_feature_names()))

    tf_idf_transformer = TfidfTransformer(use_idf=True).fit(X)
    X_tf_idf = tf_idf_transformer.transform(X)
    print(X_tf_idf.shape)