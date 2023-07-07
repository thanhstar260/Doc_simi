from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def lsa_similarity(text1, text2):
    with open(r"D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
    corpus = [line.strip() for line in lines]
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf.shape)
    svd = TruncatedSVD(5)
    lsa = make_pipeline(svd, Normalizer(copy=False, norm='l2'))
    tfidf_lsa = lsa.fit_transform(tfidf)
    print(tfidf_lsa.shape)
    similarity = cosine_similarity(tfidf_lsa[-2].reshape(1, -1), tfidf_lsa[-1].reshape(1, -1))
    return abs(similarity[0][0])

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(lsa_similarity(text1, text2))