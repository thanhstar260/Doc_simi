from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_similarity(text1, text2):
    with open("corpus.txt", "r") as file:
        # Read all lines from the file and store them in a list
        lines = file.readlines()

    # Strip any newline characters from each line
    corpus = [line.strip() for line in lines]

    # Thêm hai văn bản đã tiền xử lý vào tập dữ liệu corpus
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


with open("./corpus.txt", "r") as file:
    # Read all lines from the file and store them in a list
    lines = file.readlines()

lines = [line.strip().rstrip(".") for line in lines]
lines
text1 = lines[0]
text2 = lines[1]

print(tfidf_similarity(text1,text2))