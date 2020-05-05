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
from keras.layers import LSTM
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

tokenizer = Tokenizer(num_words=len(data.word))
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
train_data = filter_by_reviews_title(df, my_reviews)

# кодируем отзывы
print("texts_to_sequences")
test_data["text"] = tokenizer.texts_to_sequences(test_data["text"])
train_data["text"] = tokenizer.texts_to_sequences(train_data["text"])
vocab_size = len(tokenizer.word_index) + 1

# дополняем отзывы до длины в 300
print("pad_sequences")
maxlen = 600
reviews_test_prepared = pad_sequences(test_data["text"].to_numpy(), maxlen=maxlen, padding='post')
reviews_train_prepared = pad_sequences(train_data["text"].to_numpy(), maxlen=maxlen, padding='post')
print("to_categorical")
# переводим наши классы(-1 0 1) в категорийные
num_classes = 3
labels_test_prepared = keras.utils.to_categorical(test_data["label"], num_classes)
labels_train_prepared = keras.utils.to_categorical(train_data["label"], num_classes)

# создаем эмбеддинг матрицу
print("Embedding")
embedding_dim = 300
embedding_matrix = get_embedding_matrix(data, tokenizer.word_index, embedding_dim)

# имплементируем сверточную нейронку
print("Sequantial")
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

batch_size = 64
epochs = 6

# тренируем
print("Fit")
model.fit(reviews_train_prepared, labels_train_prepared,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(reviews_test_prepared, labels_test_prepared))

score = model.evaluate(reviews_train_prepared, labels_train_prepared,
                       batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

score = model.evaluate(reviews_test_prepared, labels_test_prepared,
                       batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

# предсказываем
print("Predict")
result = model.predict(reviews_test_prepared)
print(classification_report(labels_test_prepared.argmax(axis=1), result.argmax(axis=1)))
