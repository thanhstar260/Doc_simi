import math
import re
import string
import numpy as np 
from collections import Counter

def calculate_bow(document, vocabulary):
    vector = {term: document.split().count(term) for term in vocabulary}
    return vector

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[key] * vec2.get(key, 0) for key in vec1)
    magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    return dot_product / (magnitude1 * magnitude2)



def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub('[' + string.punctuation + ']', '', text)  # Loại bỏ dấu câu
    text = ' '.join(text.split()) # Xóa dấu " " dư thừa
    # text = text.split()
    return text


def calculate_similarity(text1, text2):
    # Xử lý văn bản và tạo tập từ vựng
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    corpus = [text1, text2]
    vocabulary = set()
    for document in corpus:
        vocabulary.update(document.split())
    vocabulary = sorted(vocabulary)

    # Tính vector BoW cho từng văn bản
    vector1 = calculate_bow(text1, vocabulary)
    vector2 = calculate_bow(text2, vocabulary)
        # Tính độ tương đồng bằng các phương pháp
    cosine_sim = cosine_similarity(vector1, vector2)
    return cosine_sim


def BoW(sentence1, sentence2):

    words2 = preprocess_text(sentence2)
    words1 = preprocess_text(sentence1)

    all_words = set(words1).union(set(words2))
    c1 = Counter(words1)
    c2 = Counter(words2)
    bow1 = {word: c1[word] for word in all_words}
    bow2 = {word: c2[word] for word in all_words}
    vector1 = np.array([bow1[word] if word in words1 else 0 for word in all_words])
    vector2 = np.array([bow2[word] if word in words2 else 0 for word in all_words])
    return vector1, vector2


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def BoW_levin_jaccard_cosine_similarity(text1, text2, measure):
    vector1, vector2 = BoW(text1, text2)
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    similarity_cosine = cosine_similarity(vector1, vector2)
    return similarity_cosine
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i][j - 1],      # Insert
                                  d[i - 1][j],      # Delete
                                  d[i - 1][j - 1])  # Replace
    return d[m][n]


def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub('[' + string.punctuation + ']', '', text)  # Loại bỏ dấu câu
    text = ' '.join(text.split()) # Xóa dấu " " dư thừa
    # text = text.split()
    return text

def calculate_similarity(text1, text2):

    # Xử lý văn bản và tạo tập từ vựng
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    corpus = [text1, text2]
    vocabulary = set()
    for document in corpus:
        vocabulary.update(document.split())
    vocabulary = sorted(vocabulary)
    # print(vocabulary)
    # Tính vector TF-IDF cho từng văn bản
    vector1 = calculate_bow(text1, vocabulary)
    vector2 = calculate_bow(text2, vocabulary)

    # Tính độ tương đồng bằng các phương pháp
    cosine_sim = cosine_similarity(vector1, vector2)

    return cosine_sim, levenshtein_dist, jaccard_sim

def split_paragraph(text, num_sentences_per_chunk):
    # Tách câu trong đoạn văn thành một danh sách
    sentences = text.split(". ")

    # Chia danh sách các câu thành các phần tử, mỗi phần tử chứa số lượng câu được chỉ định
    chunks = [' '.join(sentences[i:i+num_sentences_per_chunk]) for i in range(0, len(sentences), num_sentences_per_chunk)]

    return chunks

def find_most_similar_segment(text1, text2, measure, n_sentences):
    # Tiền xử lý văn bản
    # text1 = preprocess_text(text1)
    # text2 = preprocess_text(text2)
    
    # Chia văn bản thành các đoạn (mỗi đoạn có 3 câu có thể thay đổi)
    segments1 = split_paragraph(text1, n_sentences)
    segments2 = split_paragraph(text2, n_sentences)
    
    max_similarity = float('-inf')
    most_similar_segment = None
    
    # Tìm đoạn có độ tương đồng cao nhất
    for segment1 in segments1:
        for segment2 in segments2:
            if measure == "cosine":
                similarity = calculate_similarity(segment1, segment2)[0]
            if measure == "levin":
                similarity = calculate_similarity(segment1, segment2)[1]
            if measure == "jaccard":
                similarity = calculate_similarity(segment1, segment2)[-1]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_segment = (segment1,segment2) if similarity > 0 else ""
    
    return max_similarity, most_similar_segment

# Sử dụng hàm calculate_similarity để tính độ tương đồng
text1 = "love"
text2 = "ass"

# Đọc nội dung từ hai file text1.txt và text2.txt
with open('text1.txt', 'r') as file:
    text1 = file.read()

with open('text2.txt', 'r') as file:
    text2 = file.read()

cosine_sim, levenshtein_dist, jaccard_sim = calculate_similarity(text1, text2)

print("Cosine Similarity:", cosine_sim)
print("Levenshtein Distance:", levenshtein_dist)
print("Jaccard Similarity:", jaccard_sim)

similarity_measure = 'cosine'  # Chọn phương pháp đo độ tương đồng: 'cosine', 'levin', 'jaccard'
n_sentences = 1
most_similar_segment = find_most_similar_segment(text1, text2, similarity_measure, n_sentences)

print(f"Most similar segment (using {similarity_measure} similarity):")
print(most_similar_segment)