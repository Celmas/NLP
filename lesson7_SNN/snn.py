import nltk
import numpy as np
import keras
import pandas
import pymorphy2
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import GlobalMaxPool1D
from nltk.corpus import stopwords
from sklearn.metrics import classification_report


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    normalized_text = []
    tokens = nltk.word_tokenize(text)
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...']
    for token in tokens:
        if token in puncto: continue
        normalized_text.append(analyzer.parse(token)[0].normal_form)
    stops = set(stopwords.words("russian"))
    normalized_text = [w for w in normalized_text if not w in stops]
    return normalized_text


def get_embedding_matrix(dataframe, word_index, dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, dim))
    for index in range(len(dataframe["word"])):
        if dataframe["word"][index] in word_index:
            x = word_index[dataframe["word"][index]]
            embedding_matrix[x] = np.array(dataframe["value"][index], dtype=np.float32)[:dim]
    return embedding_matrix


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


# парсим вектора из готовой модели
data = pandas.read_csv('C:/Users/Celmas/Downloads/model.txt', skiprows=1, sep=r'\s{2,}', engine='python', names=[1])
data = pandas.DataFrame(data[1].str.split(r'\s{1,}', 1), columns=[1])
data = data[1].apply(pandas.Series)
data.columns = ["word", "value"]
i = 0
print("Reading model.txt")
for row in data["word"]:
    data["word"][i] = row.split("_", 1)[0]
    i += 1
i = 0
for row in data["value"]:
    data["value"][i] = row.split(" ")
    i += 1

tokenizer = Tokenizer(num_words=189193)
print("Fit on texts")
tokenizer.fit_on_texts(data["word"])

# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
print("Normalizing")
df['text'] = df['text'].map(lambda x: get_normal_form(x))
print("End Normalizing")
my_reviews = ["Гладиатор", "Начало", "Помни"]

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

# кодируем отзывы
print("texts_to_sequences")
test_data["text"] = tokenizer.texts_to_sequences(test_data["text"])
df["text"] = tokenizer.texts_to_sequences(df["text"])

# дополняем отзывы до длины в 300
print("pad_sequences")
reviews_train_prepared = pad_sequences(test_data["text"].to_numpy(), maxlen=300, padding='post')
reviews_test_prepared = pad_sequences(df["text"].to_numpy(), maxlen=300, padding='post')
print("to_categorical")
# переводим наши классы(-1 0 1) в категорийные
labels_train_prepared = keras.utils.to_categorical(test_data["label"], 3)
labels_test_prepared = keras.utils.to_categorical(df["label"], 3)

# создаем эмбеддинг матрицу
print("Embeding")
embedding_matrix = get_embedding_matrix(data, tokenizer.word_index, 300)

# имплементируем сверточную нейронку
print("Sequantial")
model = Sequential()
model.add(Embedding(141374, 300, weights=[embedding_matrix], input_length=300, trainable=False))
model.add(Conv1D(300, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(9))
model.add(Activation('softmax'))
model.compile(metrics=["accuracy"], optimizer='adam', loss='binary_crossentropy')

# тренируем
print("Fit")
# model.fit(reviews_train_prepared, labels_train_prepared, epochs=10, verbose=False)
# model.fit(reviews_train_prepared, labels_train_prepared, epochs=20, verbose=False)
model.fit(reviews_train_prepared, labels_train_prepared, epochs=30, verbose=False)

# предсказываем
print("Predict")
result = model.predict(reviews_test_prepared)
print(classification_report(labels_test_prepared.argmax(axis=1), result.argmax(axis=1)))