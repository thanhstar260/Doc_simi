import math
import re
import string

def tfidf(term, document, corpus):
    tf = document.count(term) / len(document)
    idf = math.log((len(corpus) +1)/ (sum(1 for doc in corpus if term in doc)+1)) +1
    return tf * idf

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[key] * vec2.get(key, 0) for key in vec1)
    magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    return dot_product / (magnitude1 * magnitude2)

def levenshtein_distance(str1, str2):
    if len(str1) < len(str2):
        return levenshtein_distance(str2, str1)
    if len(str2) == 0:
        return len(str1)
    previous_row = range(len(str2) + 1)
    for i, char1 in enumerate(str1):
        current_row = [i + 1]
        for j, char2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char1 != char2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

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
    vector1 = {term: tfidf(term, text1.split(), corpus) for term in vocabulary}
    vector2 = {term: tfidf(term, text2.split(), corpus) for term in vocabulary}

    # Tính độ tương đồng bằng các phương pháp
    cosine_sim = cosine_similarity(vector1, vector2)
    levenshtein_dist = levenshtein_distance(text1, text2)
    jaccard_sim = jaccard_similarity(set(text1.split()), set(text2.split()))

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