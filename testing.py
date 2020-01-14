import nltk
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import time

# Local imports

# Sample text for testing
from garbage import text

nltk.download('wordnet')
nltk.download('stopwords')


# Uzimanje tokena iz teksta
tokens = nltk.word_tokenize(text)

print(f"Number of tokens: {len(tokens)}")
print(tokens)


# Lematizacija tokena
# Pretvaranje "run" u "running" i slicno

lemmatizer = WordNetLemmatizer()

lem_tokens = []

start = time.time()

for t in tokens:
	lem_t = lemmatizer.lemmatize(t)
	if (t != lem_t):
		print(t)
		print(lemmatizer.lemmatize(t))
		print("----------")
		lem_tokens.append(lem_t)
	else :
		lem_tokens.append(t)

end = time.time()

print()
print(f"[Lemmatizing] Time elapsed: {end - start}")
print()
print()

# Stop word removal

before = len(lem_tokens)

start = time.time()

lem_tokens = [token for token in lem_tokens if token not in stopwords.words('english')]

end = time.time()

print()
print(f"[Stopword removal] Time elapsed: {end - start}" )
print()
print()


after = len(lem_tokens)

print(f"Before: {before} and after: {after}")

# Frenkvencija reci

def word_count(tokens):

	w_count = {}

	for t in tokens:
		t = t.lower()
		if t in w_count:
			w_count[t] += 1
		else:
			w_count[t] = 1

	return w_count

wc = word_count(lem_tokens)

print()
print(wc)
print()
print()

# reci krace od 2 karaktera

print("Words shorter than 3 chars\n")

for word in wc.keys():
	if len(word) <= 2:
		print(word)

# tf-idf:

