import nltk
import pandas
import pymorphy2
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords


def get_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    normalized_text = []
    tokens = nltk.word_tokenize(text)
    puncto = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '–']
    for token in tokens:
        if token in puncto: continue
        normalized_text.append(analyzer.parse(token)[0].normal_form)
    stops = set(stopwords.words("russian"))
    stops.add('фильм')
    stops.add('это')
    stops.add('весь')
    stops.add('который')
    normalized_text = [w for w in normalized_text if not w in stops]
    return normalized_text


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


def main():
    dictionary = corpora.Dictionary(df['text'])
    corpus = [dictionary.doc2bow(text) for text in df['text']]

    NUM_TOPICS = 10
    NUM_WORDS = 15
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)

    for index, topic in ldamodel.show_topics(num_topics=NUM_TOPICS, formatted=False, num_words=NUM_WORDS):
        print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))

    cm = CoherenceModel(model=ldamodel, texts=df['text'], dictionary=dictionary)
    coherence = cm.get_coherence()
    print(coherence)


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
df['text'] = df['text'].map(lambda x: get_normal_form(x))

# из-за проблемы многопоточности в windows
if __name__ == "__main__":
    main()