import nltk as nltk
import pandas
import pymorphy2
import numpy as np
import gensim.models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1
    return review_feature_vecs


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    normalized_text = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        normalized_text.append(analyzer.parse(token)[0].normal_form)
    return normalized_text


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
df['text'] = df['text'].map(lambda x: get_normal_form(x))
my_reviews = ["Гладиатор", "Начало", "Помни"]

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

# Создаем модель word2vec
model = gensim.models.Word2Vec(sentences=df['text'], min_count=2, iter=200)

# Синонимы
print(model.wv.most_similar(positive=['фильм'], topn=5))
print(model.wv.most_similar(positive=['мультфильм'], topn=5))
print(model.wv.most_similar(positive=['актёр'], topn=5))
print(model.wv.most_similar(positive=['девочка'], topn=5))
print(model.wv.most_similar(positive=['любовь'], topn=5))

# Получаем вектора
trainDataVecs = get_avg_feature_vecs(df['text'], model, 100)
testDataVecs = get_avg_feature_vecs(test_data['text'], model, 100)

# Тренируем модель
rfc = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
rfc.fit(trainDataVecs, df['label'])

# Применяем модель к текстовым текстам
predicted = rfc.predict(testDataVecs)

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