import csv
import spacy
nlp = spacy.load('en_core_web_sm')

rows = []

with open('princess_corpus.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row = dict(row)
        rows.append(row)


for row in rows:
    doc = nlp(row['Text'])
    for token in doc:
        row['tokens'] = dict()
        if token.pos_ != 'PUNCT' and token.is_stop is False:
            tokens = dict()
            tokens['pos'] = token.pos_
            tokens['lemma'] = token.lemma_
            row['tokens'][token] = tokens
            print(row)


