
# code chay
def generate_ngrams(text, n):
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def ngram_similarity(text1, text2, n):
    ngrams1 = set(generate_ngrams(text1, n))
    ngrams2 = set(generate_ngrams(text2, n))
    common_ngrams = ngrams1.intersection(ngrams2)
    
    similarity = len(common_ngrams) / float(max(len(ngrams1), len(ngrams2)))
    return similarity

# code sử dụng thư viện
# from nltk import ngrams
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def ngram_similarity(text1, text2,n_gram = 3):
#     corpus = [text1, text2]
#     if n_gram > 1:
#         # Tạo danh sách các n-gram cho cả hai đoạn văn bản
#         ngrams_corpus = []
#         for doc in corpus:
#             grams = [' '.join(gram) for gram in ngrams(doc.split(), n_gram)]
#             ngrams_corpus.append(' '.join(grams))
#         corpus = ngrams_corpus

#     vectorizer = CountVectorizer()
#     vectorized_corpus = vectorizer.fit_transform(corpus)
#     similarity_matrix = cosine_similarity(vectorized_corpus)
#     similarity = similarity_matrix[0, 1]
#     return similarity

text1 = 'I have a dog and I love it'
text2 = 'I have a cat and I hate it'
n = 3

similarity_score = ngram_similarity(text1, text2, n)
print("Similarity Score:", similarity_score)