import nltk
import numpy
import pandas
import pymorphy2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%', '&']
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


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ["Гладиатор", "Начало", "Помни"]

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

# Получаем словарь всех слов и представление текстов в виде мешка слов
vocabulary, bag_of_words = get_vocab_and_bag_of_words_vectors(df.values, [])
# Подготавливаем тестовое представление текстов
vocabulary, test_bag_of_words = get_vocab_and_bag_of_words_vectors(test_data.values, vocabulary)

# Тренируем модель
print("---Train---")
rfc = RandomForestClassifier(max_depth=20)
rfc.fit(bag_of_words, df['label'])

# Применяем модель к текстовым текстам
print("---Predict---")
predicted = rfc.predict(test_bag_of_words)

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
