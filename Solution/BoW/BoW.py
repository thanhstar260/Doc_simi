import math
import re
import string
import numpy as np 
from collections import Counter

def bow_vec(document, vocabulary):
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

def bow_similarity(text1, text2):
    # Xử lý văn bản và tạo tập từ vựng
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    corpus = [text1, text2]
    vocabulary = set()
    for document in corpus:
        vocabulary.update(document.split())
    vocabulary = sorted(vocabulary)
    vector1 = bow_vec(text1, vocabulary)
    vector2 = bow_vec(text2, vocabulary)
        # Tính độ tương đồng bằng các phương pháp
    cosine_sim = cosine_similarity(vector1, vector2)
    return cosine_sim

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(bow_similarity(text1, text2))