def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Example usage
text = "The quick brown fox jumps over the lazy dog"
n = 2

result = generate_ngrams(text, n)
print(result)
