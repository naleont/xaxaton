import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("en_core_web_sm")

stops = stopwords.words("english")
tfidf = TfidfVectorizer(
    analyzer="word",
    stop_words=stops
)

def tokenizer(texts):
    tokenized_texts = []
    for text in texts:
        doc = nlp(text[0])
        tokenized_text = ' '.join([token.lemma_ for token in doc])
        tokenized_texts.append(tokenized_text)
    return tokenized_texts


ds = pd.read_csv('dataset.csv')
for i in ds.columns[:-1]:
    grouped = ds.groupby(i).agg({'Text': lambda x: ' . '.join(x)})
    tokenized_texts = tokenizer(grouped.values)
    categories = grouped.axes[0]
    texts_tfidf = tfidf.fit_transform(tokenized_texts)
    pca = PCA(n_components=2)
    texts_tfidf = texts_tfidf.todense()
    coords = pca.fit_transform(texts_tfidf)
    plt.scatter(coords[:, 0], coords[:, 1], color='red')
    plt.title(i)
    for j, category in enumerate(categories):
        plt.annotate(category, xy=(coords[j, 0], coords[j, 1]))
    plt.show()