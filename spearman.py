''' Dataset for Sematic Textual Similarity task '''
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import nltk
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_sm')
dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
df = pd.DataFrame(dataset)
array = df.values
'''
array([['A man with a hard hat is dancing.',
        'A man wearing a hard hat is dancing.', 5.0],
       ['A young child is riding a horse.', 'A child is riding a horse.',
        4.75],
       ['A man is feeding a mouse to a snake.',
        'The man is feeding a mouse to the snake.', 5.0],
       ...,
       ['Volkswagen skids into red in wake of pollution scandal',
        'Volkswagen\'s "gesture of goodwill" to diesel owners', 2.0],
       ['Obama is right: Africa deserves better leadership',
        'Obama waiting for midterm to name attorney general', 0.0],
       ['New video shows US police officers beating mentally-ill man',
        'New York police officer critically wounded in hatchet attack',
        0.0]], dtype=object)
'''

'''
Divide this array into:
1st and 2nd column: sentence_pairs -> test set to compare multiple metrics
3rd column: scores -> groundtruth score of a pair (a row in sentence_pairs) on the scale of [0:5]
'''
sentence_pairs = array[:,:2]
scores = array[:,2]

# @@@ Dưới đây là một số metric để tính similarity, và sau đó tính hệ số tương quan hạng Spearman @@@

'''---------------Word2vec-------------'''
word2vec = KeyedVectors.load_word2vec_format(r'Doc2Vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
def word2vec_similarity(text1 :str, text2: str) -> float:
    text1 = nlp(text1.lower())
    text2 = nlp(text2.lower())
    text1 = [token.lemma_ for token in text1 if not token.is_stop and not token.is_punct]
    text2 = [token.lemma_ for token in text2 if not token.is_stop and not token.is_punct]
    vector1 = np.mean([word2vec[word] for word in text1 if word in word2vec.key_to_index], axis=0)
    vector2 = np.mean([word2vec[word] for word in text2 if word in word2vec.key_to_index], axis=0)
    cosine_similarity_value = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return float(cosine_similarity_value)

'''----------------LSA-----------------'''
with open(r"TF_IDF\corpus.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()
    corpus = [line.strip() for line in lines]

def lsa_similarity(text1 :str, text2 :str) -> float:
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(70)
    lsa = make_pipeline(svd, Normalizer(copy=False, norm='l2'))
    tfidf_lsa = lsa.fit_transform(tfidf)
    similarity = cosine_similarity(tfidf_lsa[-2].reshape(1, -1), tfidf_lsa[-1].reshape(1, -1))
    return abs(similarity[0][0])

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
    if len(s) == 0:
        return 0  # return a default value when no matches found
    else:
        return sum(s)/len(s)

def document_path_similarity(doc1, doc2):
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

'''
@@@ Hàm tính vector similarity dựa trên ma trận sentence_pairs @@@

---Input---
sentence_pairs(np.array)  -> ma trận định nghĩa ở trên, mỗi hàng có 2 câu cần tính similarity
metric_function(Callable) -> hàm dùng để tính similarity của sentence_pairs theo từng hàng
---Output---
result(np.array)          -> vector chứa similarity của từng cặp câu ở từng hàng tương ứng

'''
from typing import Callable

def similarity_vector(sentence_pairs : np.array, metric_function : Callable) -> np.array:
  return np.vectorize(metric_function)(sentence_pairs[:,0],sentence_pairs[:,1])

'''
Hàm tính Spearman correlation (có giá trị chạy từ -1 đến 1, tuy nhiên metric hiển nhiên đồng biến
với groundtruth nên trong bài toán này hệ số luôn có giá trị >=0) giữa 2 vector
'''
def spearman_rank_correlation(vector1 : np.array, vector2 : np.array) -> float:
    # Tính Spearman rank correlation
    correlation, _ = spearmanr(vector1, vector2)

    return correlation

'''Tính toán Spearman correlation giữa word2vec và LSA, correlation càng gần 1 thì metric càng tốt'''
word2vec = similarity_vector(sentence_pairs,word2vec_similarity)
lsa      = similarity_vector(sentence_pairs,lsa_similarity)
synset = similarity_vector(sentence_pairs, document_path_similarity)

print('Spearman between Word2vec and groundtruth: ',spearman_rank_correlation(word2vec,scores))
print('Spearman between LSA and groundtruth: ',spearman_rank_correlation(lsa,scores))
print('Spearman between Synset and groundtruth: ', spearman_rank_correlation(synset, scores))

'''
Output:
Spearman between Word2vec and groundtruth:  0.7644742056095604
Spearman between LSA and groundtruth:  0.6279419324105496
'''