import string

import nltk
import numpy
import pandas
import pymorphy2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in string.punctuation: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_vocab_and_bag_of_words_vectors(all_texts, vocab):
    normalized_texts = []
    for elem in all_texts:
        normalized_texts.append(get_normal_form(elem[1]))
    if len(vocab) == 0:
        vocab = numpy.unique(numpy.concatenate(normalized_texts))
    vectors = []
    j = 0
    for text in normalized_texts:
        bag_vector = numpy.zeros(len(vocab))
        for w in text:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
        vectors.append(bag_vector)
        j += 1
        print("iteration: " + str(j))
    return vocab, vectors


def filter_by_reviews_title(data_frame, reviews_titles):
    filtered = data_frame[~data_frame['title'].isin(reviews_titles)]
    return filtered


def get_pos(text):
    analyzer = pymorphy2.MorphAnalyzer()
    tokens = nltk.word_tokenize(text)
    noun = 0
    adj = 0
    verb = 0
    advb = 0
    for token in tokens:
        if token in string.punctuation: continue
        pos = analyzer.parse(token)[0].tag.POS
        if pos == 'NOUN':
            noun = noun + 1
        elif pos == 'ADJF' or pos == 'ADJS':
            adj = adj + 1
        elif pos == 'VERB' or pos == 'INFN':
            verb = verb + 1
        elif pos == 'ADVB':
            advb = advb + 1
    return [noun, adj, verb, advb]


def get_pos_for_texts(all_texts):
    poses1 = []
    for text in all_texts:
        poses1.append(get_pos(text))
    return poses1


def get_puct(text):
    tokens = nltk.word_tokenize(text)
    puct_vector = numpy.zeros(len(string.punctuation))
    for token in tokens:
        if token in string.punctuation:
            for i, puncto in enumerate(string.punctuation):
                if puncto == token:
                    puct_vector[i] += 1
    return puct_vector


def get_puct_for_texts(all_texts):
    puct1 = []
    for text in all_texts:
        puct1.append(get_puct(text))
    return puct1


def print_stat(predicted, test_data):
    # Находим Accuracy
    true_positives = 0
    k = 0
    for prediction in predicted:
        if prediction == test_data.values[k][2]:
            true_positives += 1
        k += 1

    print("Accuracy: {}".format(true_positives / len(predicted)))
    # Находим Precision, Recall, Fscore
    precision_recall_fscore = precision_recall_fscore_support(test_data['label'].values, predicted)
    print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
    print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
    print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ["Гладиатор", "Начало", "Помни"]

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

# Получаем словарь всех слов и представление текстов в виде мешка слов
vocabulary, bag_of_words = get_vocab_and_bag_of_words_vectors(df.values, [])
# Обнаруживаем части речи и получаем масив векторов
poses = get_pos_for_texts(df.values)
# Получаем вектор знаков препинаний
pucto = get_puct_for_texts(df.values)

# Подготавливаем тестовое представление текстов
vocabulary, test_bag_of_words = get_vocab_and_bag_of_words_vectors(test_data.values, vocabulary)
# Обнаруживаем части речи и получаем масив векторов для тестовой выборки
test_poses = get_pos_for_texts(test_data.values)
# Получаем вектор знаков препинаний
test_pucto = get_puct_for_texts(test_data.values)

# Тренируем модель
print("---Train for all---")
rfc = RandomForestClassifier(max_depth=20)
rfc.fit(poses.append(bag_of_words).append(pucto), df['label'])

# Применяем модель к текстовым текстам
print("---Predict---")
predicted = rfc.predict(test_poses.append(test_bag_of_words).append(test_pucto))

print_stat(predicted, test_data)

# Тренируем модель
print("---Train without bow---")
rfc = RandomForestClassifier(max_depth=20)
rfc.fit(poses.append(pucto), df['label'])

# Применяем модель к текстовым текстам
print("---Predict---")
predicted = rfc.predict(test_poses.append(test_pucto))

print_stat(predicted, test_data)

# Тренируем модель
print("---Train without pos---")
rfc = RandomForestClassifier(max_depth=20)
rfc.fit(pucto.append(bag_of_words), df['label'])

# Применяем модель к текстовым текстам
print("---Predict---")
predicted = rfc.predict(test_pucto.append(test_bag_of_words))

print_stat(predicted, test_data)

# Тренируем модель
print("---Train without pucto---")
rfc = RandomForestClassifier(max_depth=20)
rfc.fit(poses.append(bag_of_words), df['label'])

# Применяем модель к текстовым текстам
print("---Predict---")
predicted = rfc.predict(test_poses.append(test_bag_of_words))

print_stat(predicted, test_data)
