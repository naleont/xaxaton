import pandas as pd

from nltk.corpus import stopwords
stops = stopwords.words('english')
stops.extend(["i'm", "he's", "i've", "i'll"])

# читска от пунктуации
punct_list = '!"«»“”#$%&\–-–—()*+,./\:;<=>?@[]^_`{|}~1234567890'

def clean(text):
    text = text.lower()
    for char in text:
        if char in punct_list:
            text = text.replace(char, '')
    return text

# функция находящая полезные слова
def extract_words(dataset):
    df = pd.read_csv(dataset)
    quotes = df['Text'].values
    
    for quot in quotes:
        words = clean(quot).split()
        good_words = []
        for word in words:
            if word not in stops:
                good_words.append(word)
                
    return good_words
