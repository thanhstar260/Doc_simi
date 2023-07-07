def generate_ngrams(text, n):
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calculate_similarity(text1, text2, n):
    ngrams1 = set(generate_ngrams(text1, n))
    ngrams2 = set(generate_ngrams(text2, n))
    common_ngrams = ngrams1.intersection(ngrams2)
    
    similarity = len(common_ngrams) / float(max(len(ngrams1), len(ngrams2)))
    return similarity

text1 = 'I have a dog and I love it'
text2 = 'I have a cat and I hate it'
n = 3

similarity_score = calculate_similarity(text1, text2, n)
print("Similarity Score:", similarity_score)