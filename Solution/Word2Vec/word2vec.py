from fastapi import FastAPI, Form
import uvicorn
from sklearn.metrics import jaccard_score
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
import spacy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk import ngrams

# Load Doc2Vec model
nlp = spacy.load('en_core_web_sm')
word2vec = KeyedVectors.load_word2vec_format(r'GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

def word2vec_similarity(text1, text2):
    text1 = nlp(text1.lower())
    text2 = nlp(text2.lower())
    text1 = [token.lemma_ for token in text1 if not token.is_stop and not token.is_punct]
    text2 = [token.lemma_ for token in text2 if not token.is_stop and not token.is_punct]
    vector1 = np.mean([word2vec[word] for word in text1 if word in word2vec.key_to_index], axis=0)
    vector2 = np.mean([word2vec[word] for word in text2 if word in word2vec.key_to_index], axis=0)
    print(vector1.shape)
    print(vector2.shape)
    cosine_similarity_value = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return float(cosine_similarity_value)

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
word2vec_similarity(text1, text2)