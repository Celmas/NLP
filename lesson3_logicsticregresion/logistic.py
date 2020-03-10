import nltk
import numpy
import pandas
import pymorphy2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in puncto: continue
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


df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ["Гладиатор", "Начало", "Помни"]
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

vocabulary, bag_of_words = get_vocab_and_bag_of_words_vectors(df.values, [])
vocabulary, test_bag_of_words = get_vocab_and_bag_of_words_vectors(test_data.values, vocabulary)

print("---Train---")
reg = LogisticRegression(max_iter=10000)
reg.fit(bag_of_words, df['label'])

print("---Predict---")
predicted = reg.predict(test_bag_of_words)

true_positives = 0
k = 0
for prediction in predicted:
    if prediction == test_data.values[k][2]:
        true_positives += 1
    k += 1

print("Accuracy: {}".format(true_positives / len(predicted)))
precision_recall_fscore = precision_recall_fscore_support(test_data['label'].values, predicted)
print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")

dict_of_negative = dict(zip(vocabulary, reg.coef_[0]))
dict_of_neutral = dict(zip(vocabulary, reg.coef_[1]))
dict_of_positive = dict(zip(vocabulary, reg.coef_[2]))

sorted_negative_dictionary = {k: v for k, v in sorted(dict_of_negative.items(), key=lambda item: item[1], reverse=True)}
sorted_neutral_dictionary = {k: v for k, v in sorted(dict_of_neutral.items(), key=lambda item: item[1], reverse=True)}
sorted_positive_dictionary = {k: v for k, v in sorted(dict_of_positive.items(), key=lambda item: item[1], reverse=True)}

negative = open("negatives.txt", "w", encoding="utf-8")
neutral = open("neutrals.txt", "w", encoding="utf-8")
positive = open("positives.txt", "w", encoding="utf-8")

print(sorted_negative_dictionary, file=negative)
print(sorted_neutral_dictionary, file=neutral)
print(sorted_positive_dictionary, file=positive)
