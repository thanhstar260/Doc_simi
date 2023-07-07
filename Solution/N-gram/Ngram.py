import math
import re
import string
import numpy as np

def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub('[' + string.punctuation + ']', '', text)  # Loại bỏ dấu câu
    text = ' '.join(text.split()) # Xóa dấu " " dư thừa
    # text = text.split()
    return text

def calculate_ngram(text, n):
    words = text.split()  # Tách câu thành các từ
    # Tạo danh sách các n-gram
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.append(ngram)

    return ngrams

def ngram_vec(document, vocabulary):
    vector = {term: document.split().count(term) for term in vocabulary}
    return vector

def cosine_similarity(dict1, dict2):
    # Compute dot product
    dot_product = 0
    for key in dict1:
        if key in dict2:
            dot_product += dict1[key] * dict2[key]
    
    # Compute vector magnitudes
    magnitude_dict1 = math.sqrt(sum(val ** 2 for val in dict1.values()))
    magnitude_dict2 = math.sqrt(sum(val ** 2 for val in dict2.values()))
    
    # Handle edge case where one of the dictionaries is empty
    if magnitude_dict1 == 0 or magnitude_dict2 == 0:
        return 0
    
    # Compute cosine similarity
    similarity = dot_product / (magnitude_dict1 * magnitude_dict2)
    return similarity

def ngram_similarity(text1, text2,n_gram = 1):
    # Xử lý văn bản và tạo tập từ vựng
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    corpus = [text1, text2]
    if n_gram == 1 :
        vocabulary = set()
        for document in corpus:
            vocabulary.update(document.split())
        vocabulary = sorted(vocabulary)
    else:
        n_gram1 = calculate_ngram(text1,n_gram)
        n_gram2 = calculate_ngram(text2,n_gram)
        vocabulary = set(n_gram1 + n_gram2)
    # Tính vector ngram cho từng văn bản
    vector1 = ngram_vec(text1, vocabulary)
    vector2 = ngram_vec(text2, vocabulary)
        # Tính độ tương đồng bằng các phương pháp
    cosine_sim = cosine_similarity(vector1, vector2)
    return cosine_sim

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(ngram_similarity(text1, text2, 1))