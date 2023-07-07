from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import Normalizer
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import spacy
# nlp = spacy.load('en_core_web_sm')

# Load Doc2Vec model
word2vec = KeyedVectors.load_word2vec_format(r'D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/Doc2Vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

# def word2vec_similarity(text1, text2):
#     text1 = nlp(text1.lower())
#     text2 = nlp(text2.lower())
#     text1 = [token.lemma_ for token in text1 if not token.is_stop and not token.is_punct]
#     text2 = [token.lemma_ for token in text2 if not token.is_stop and not token.is_punct]
#     vector1 = np.mean([word2vec[word] for word in text1 if word in word2vec.key_to_index], axis=0)
#     vector2 = np.mean([word2vec[word] for word in text2 if word in word2vec.key_to_index], axis=0)
#     cosine_similarity_value = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
#     return float(cosine_similarity_value)


word1 = 'king'
word2 = 'man'
word3 = 'woman'
word4 = 'queen'
vectora = word2vec[word1] - word2vec[word2] + word2vec[word3]
vectorb = word2vec[word4]
cosine_similarity_value = cosine_similarity(vectora.reshape(1, -1), vectorb.reshape(1, -1))[0][0]
print("Cosine similarity between vector a and vector b:", cosine_similarity_value)


# Perform PCA for vectora
pca_a = PCA(n_components=3)
reduced_vectora = pca_a.fit_transform(vectora.reshape(1, -1))

# Perform PCA for vectorb
pca_b = PCA(n_components=3)
reduced_vectorb = pca_b.fit_transform(vectorb.reshape(1, -1))

# Plot the reduced vectors in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_vectora[:, 0], reduced_vectora[:, 1], reduced_vectora[:, 2], label='vector a')
ax.scatter(reduced_vectorb[:, 0], reduced_vectorb[:, 1], reduced_vectorb[:, 2], label='vector b')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
plt.show()
