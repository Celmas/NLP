import numpy
import pandas
import nltk
import pymorphy2


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in puncto: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_predictions(review):
    count_of_unique_words = len(unique_words)

    # нормализуем отзыв
    normalized_review = get_normal_form(review)

    positive_vector = []
    negative_vector = []
    neutral_vector = []

    # считаем один рз делители
    positive_divider = (len(positive_words) + count_of_unique_words)
    negative_divider = (len(negative_words) + count_of_unique_words)
    neutral_divider = (len(neutral_words) + count_of_unique_words)

    # для каждго слова из отзыва строим вектор вероятностей
    for word in normalized_review:
        count_in_positive = positive_words.count(word)
        positive_vector.append((count_in_positive + 1) / positive_divider)

        count_in_neutral = neutral_words.count(word)
        neutral_vector.append((count_in_neutral + 1) / neutral_divider)

        count_in_negative = negative_words.count(word)
        negative_vector.append((count_in_negative + 1) / negative_divider)

    positive_prediction = numpy.log(plus)
    neutral_prediction = numpy.log(zero)
    negative_prediction = numpy.log(minus)

    # находим сумму логарифмов, по абсолютному значению будут отражать принадлежность как и вероятность
    for multiplier in positive_vector:
        positive_prediction += numpy.log(multiplier)
    for multiplier in neutral_vector:
        neutral_prediction += numpy.log(multiplier)
    for multiplier in negative_vector:
        negative_prediction += numpy.log(multiplier)

    return [negative_prediction, neutral_prediction, positive_prediction]


df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ["Гладиатор\n", "Начало", "Помни"]

# positive = open("positives.txt", "w", encoding="utf-8")
# neutral = open("neutrals.txt", "w", encoding="utf-8")
# negative = open("negatives.txt", "w", encoding="utf-8")
#
# for value in df.values:
#     if value[0] in my_reviews: continue
#     if value[2] == "1":
#         for word in get_normal_form(value[1]):
#             positive.write(word + " ")
#     if value[2] == "0":
#         for word in get_normal_form(value[1]):
#             neutral.write(word + " ")
#     if value[2] == "-1":
#         for word in get_normal_form(value[1]):
#             negative.write(word + " ")

positive = open("positives.txt", "r", encoding="utf-8")
neutral = open("neutrals.txt", "r", encoding="utf-8")
negative = open("negatives.txt", "r", encoding="utf-8")

positive_words = positive.read().split(" ")
neutral_words = neutral.read().split(" ")
negative_words = negative.read().split(" ")
# уникальныы слова
unique_words = numpy.unique(positive_words + negative_words + neutral_words)
# p(c)
plus = minus = zero = 1 / 3

index = 0
correct_prediction = 0
for value in df.values:
    if value[0] in my_reviews:
        index += 1
        predictions = get_predictions(value[1])
        predicted_value = 0
        if (predictions[0] > predictions[1]) & (predictions[0] > predictions[2]):
            predicted_value = -1
        if (predictions[1] > predictions[0]) & (predictions[1] > predictions[2]):
            predicted_value = 0
        if (predictions[2] > predictions[0]) & (predictions[2] > predictions[1]):
            predicted_value = 1
        # выводим таблицу результатов
        print(f"{index}: {predictions[0]:.8f}, {predictions[1]:.8f}, {predictions[2]:.8f}, predicted: {predicted_value}, real: {value[2]}")
        if f"{predicted_value}" == value[2]:
            correct_prediction += 1
print("Accuracy : {}".format(correct_prediction / index))
