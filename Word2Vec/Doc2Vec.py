from gensim.models import Doc2Vec
import numpy as np
from scipy.spatial.distance import cosine

# Load Doc2Vec model
model = Doc2Vec.load(r"D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/Doc2Vec/doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model")

# Get user input
text1 = 'I have a cat'
text2 = 'I have a cat'

# Preprocess the texts
text1 = text1.lower().split()
text2 = text2.lower().split()

# Calculate the vector of each text
vector1 = model.infer_vector(text1)
vector2 = model.infer_vector(text2)

# Calculate cosine similarity
cosine_similarity = 1 - cosine(vector1, vector2)

# Print the cosine similarity and the vectors
print("Cosine similarity: ", cosine_similarity)
print("Vector representation of first text: ", vector1)
print("Vector representation of second text: ", vector2)
