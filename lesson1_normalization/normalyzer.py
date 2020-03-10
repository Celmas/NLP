import nltk
import pymorphy2
from collections import Counter
import math
import numpy


def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / float(len(text))
        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus) / sum([1.0 for i in corpus if word in i]))

    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list


f = open("reviews.txt", encoding="UTF-8")
g = open("out.txt", "w", encoding="UTF-8")
analyzer = pymorphy2.MorphAnalyzer()
puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...']
words = []
texts = f.read().replace('\n', ' ').split("SPLIT")
normalized_texts = []
print(len(texts))
for text in texts:
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in puncto: continue
        word = analyzer.parse(token)[0]
        normalized_words.append(word.normal_form)
    normalized_texts.append(normalized_words)
tfidf = compute_tfidf(normalized_texts)
print(tfidf)
#TF-IDF for every word
for dictionary in tfidf:
    sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_dictionary, file=g)

#Sumarise of all unique words and sort
all_unique_words = []
for dictionary in tfidf:
    for key in dictionary.keys():
        all_unique_words.append(key)
unique_words = numpy.unique(all_unique_words)
print(len(numpy.unique(all_unique_words)))
unique_dictionary = {}
for word in unique_words:
    unique_dictionary[word] = 0
    for dictionary in tfidf:
        unique_dictionary[word] += dictionary.get(word, 0)
sorted_dictionary = {k: v for k, v in sorted(unique_dictionary.items(), key=lambda item: item[1], reverse=True)}
print(sorted_dictionary, file=g)