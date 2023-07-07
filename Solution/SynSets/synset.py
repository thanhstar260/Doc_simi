import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

def convert_tag(tag):
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

def doc_to_synsets(doc):
    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]
    ans = list(zip(tokens,wntag))
    sets = [wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
    return final


def similarity_score(s1, s2):
    s =[]
    for i1 in s1:
        r = []
        scores = [x for x in [i1.path_similarity(i2) for i2 in s2] if x is not None]
        if scores:
            s.append(max(scores))
    return sum(s)/len(s)


def document_path_similarity(doc1, doc2):

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
doc1 = "king"
doc2 = "queen"
print(document_path_similarity(doc1, doc2))