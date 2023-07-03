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

nlp = spacy.load('en_core_web_sm')
app = FastAPI()
app.mount("/static", StaticFiles(directory=r"D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/static"), name="static")
word2vec = KeyedVectors.load_word2vec_format(r'D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/Doc2Vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

@app.get("/", response_class=HTMLResponse)
async def index():
        with open('static/main/main.html') as f:
            content = f.read()
        return content

def bow_similarity(text1, text2):
    corpus = [text1, text2]
    vectorizer = CountVectorizer()
    vectorized_corpus = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectorized_corpus)
    similarity = similarity_matrix[0, 1]
    return similarity

@app.get("/BOW", response_class=HTMLResponse)
async def compare_form_bow():
    with open('static/Typeface/BoW/BOW.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_bow")
async def compare_texts_bow(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = bow_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

def tfidf_similarity(text1, text2):
    with open("TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()

    corpus = [line.strip() for line in lines]
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-2], tfidf_matrix[-1])[0][0]
    return cosine_sim

@app.get("/TFIDF", response_class=HTMLResponse)
async def compare_form_tfidf():
    with open('static/Typeface/TF-IDF/TF-IDF.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_tfidf")
async def compare_texts_tfidf(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = tfidf_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

def lsa_similarity(text1, text2):
    with open("TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
    corpus = [line.strip() for line in lines]
    corpus.append(text1)
    corpus.append(text2)  
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    vectorized_corpus = vectorizer.fit_transform(corpus)
    lsa = TruncatedSVD(n_components=200)  
    lsa_corpus = lsa.fit_transform(vectorized_corpus)
    similarity_matrix = cosine_similarity(lsa_corpus)
    similarity = similarity_matrix[-1, -2]  
    return similarity

@app.get("/LSA", response_class=HTMLResponse)
async def compare_form_lsa():
    with open('static/Semantic/LSA/LSA.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_lsa")
async def compare_texts_lsa(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = lsa_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

def word2vec_similarity(text1, text2):
    text1 = nlp(text1.lower())
    text2 = nlp(text2.lower())
    text1 = [token.lemma_ for token in text1 if not token.is_stop and not token.is_punct]
    text2 = [token.lemma_ for token in text2 if not token.is_stop and not token.is_punct]
    vector1 = np.mean([word2vec[word] for word in text1 if word in word2vec.key_to_index], axis=0)
    vector2 = np.mean([word2vec[word] for word in text2 if word in word2vec.key_to_index], axis=0)
    cosine_similarity_value = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return float(cosine_similarity_value)

@app.get("/word2vec", response_class=HTMLResponse)
async def compare_form_word2vec():
    with open('static/Semantic/Word2Vec/W2V.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_word2vec")
async def compare_texts_word2vec(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = word2vec_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

def jaccard_similarity(text1,text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

@app.get("/Jaccard", response_class=HTMLResponse)
async def compare_form_jaccard():
    with open('static/Typeface/Jaccard/Jaccard.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_jaccard")
async def compare_texts_jaccard(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = jaccard_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)
