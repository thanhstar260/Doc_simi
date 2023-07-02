from fastapi import FastAPI, Form
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.mount("/static", StaticFiles(directory=r"D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/static"), name="static")

def calculate_similarity(text1, text2):
    corpus = [text1, text2]
    vectorizer = CountVectorizer(min_df=0, binary=True)
    vectorized_corpus = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectorized_corpus)
    similarity = similarity_matrix[0, 1]
    return similarity

@app.get("/BOW", response_class=HTMLResponse)
async def compare_form():
    with open('static/Typeface/BoW/BoW.html', 'r') as f:
        content = f.read()
    return content

@app.post("/compare")
async def compare_texts(doc1: str = Form(...), doc2: str = Form(...)) -> dict:
    similarity = calculate_similarity(doc1, doc2)
    return {"similarity": round(similarity, 4)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)
