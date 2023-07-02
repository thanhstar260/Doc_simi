from fastapi import FastAPI, Form
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()
app.mount("/static", StaticFiles(directory=r"D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/static"), name="static")

def bow_similarity(text1, text2):
    corpus = [text1, text2]
    vectorizer = CountVectorizer(min_df=0, binary=True)
    vectorized_corpus = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectorized_corpus)
    similarity = similarity_matrix[0, 1]
    return similarity

@app.get("/BOW", response_class=HTMLResponse)
async def compare_form_bow():
    with open('static/Typeface/BoW/BoW.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare_bow")
async def compare_texts_bow(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = bow_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

def tfidf_similarity(text1, text2):
    # Khởi tạo corpus
    with open("./TF_IDF/corpus.txt", "r") as file:
        lines = file.readlines()

    corpus = [line.strip() for line in lines]

    # Thêm hai văn bản đã xử lý vào tập dữ liệu corpus
    corpus.append(text1)
    corpus.append(text2)

    # Sử dụng TF-IDF Vectorizer để chuyển đổi văn bản thành ma trận TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Tính độ tương đồng cosine giữa hai văn bản đã tiền xử lý
    cosine_sim = cosine_similarity(tfidf_matrix[-2], tfidf_matrix[-1])[0][0]
    
    # Xóa hai văn bản đã thêm vào tập dữ liệu corpus
    corpus.pop()
    corpus.pop()
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)
