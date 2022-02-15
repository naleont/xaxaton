import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def quotes_order(dataset):
    df = pd.read_csv(dataset)
    quotes = df['Text'].values
    w_orders = []
    for sent in quotes:
        w_orders.append(word_order(sent))
    return w_orders


def word_order(sent):
    d = nlp(sent)
    roots = [token for token in d if token.head == token]
    orders = []
    for root in roots:
        d = {}
        d['V'] = root.i
        for child in root.children:
            if child.dep_ == 'nsubj':
                d['S'] = child.i
            if child.dep_== 'dobj':
                d['O'] = child.i
        listt = sorted(d.keys(),key=d.get)
        order = ''.join(listt)
        orders.append(order)
    return orders

