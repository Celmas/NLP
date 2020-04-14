from collections import Counter
import math
import tensorflow
from itertools import islice
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K

import nltk
import numpy
import pandas
import pymorphy2


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%', '&']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in puncto: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_notmalized_texts(data):
    normalized_texts = []
    i = 0
    for elem in data.values:
        print("iteration: {}".format(i))
        normalized_texts.append(get_normal_form(elem[1]))
        i = i + 1
    return normalized_texts


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


def compute_frequency(all_text):
    frequency = {}
    for text in all_text:
        for word in text:
            count = frequency.get(word, 0)
            frequency[word] = count + 1
    return frequency


def get_tfidf_vector(tfidf_list, frequency):
    vectors = []
    j = 0
    for text in tfidf_list:
        tfidf_vector = numpy.zeros(len(frequency))
        for w in list(text):
            for i, word in enumerate(list(frequency)):
                if word == w:
                    tfidf_vector[i] = text.get(w, 0)
        vectors.append(tfidf_vector)
        j += 1
        print("iteration: " + str(j))
    return vectors


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ["Гладиатор", "Начало", "Помни"]

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

print("---Begin normalization---")
normalized_texts_train = get_notmalized_texts(df)
normalized_texts_predict = get_notmalized_texts(test_data)
print("---End normalization---")

print("---Begin computing tfidf---")
tfidf_train = compute_tfidf(normalized_texts_train)
tfidf_predict = compute_tfidf(normalized_texts_predict)
print("---End computing tfidf---")

freq = compute_frequency(normalized_texts_train)
freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
freq = dict(islice(freq.items(), 0, 500))

print("---Begin getting vectors---")
x_train_vector = get_tfidf_vector(tfidf_train, freq)
y_train_vector = get_tfidf_vector(tfidf_predict, freq)
print("---End getting vectors---")

model = Sequential()
model.add(Dense(512, input_shape=(500,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
print("---TRAINING---")
model.fit(numpy.array(x_train_vector), numpy.array(df[['label']]), epochs=10, batch_size=32)
print("---PREDICT---")
loss, accuracy, f1_score, precision, recall = model.evaluate(y_train_vector, test_data['label'], verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)
print("F1:", f1_score)
print("Precision:", precision)
print("Recall:", recall)


