import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import spacy
import networkx as nx
import matplotlib.pyplot as plt

'''Chay tren local thi dung pip tai thu vien nltk, spacy, networkx
chay them lenh python -m spacy download en_core_web_sm'''
# Tải dữ liệu WordNet (nếu chưa tải)
# nltk.download('wordnet')

# Khởi tạo bộ phân tích từ vựng của SpaCy
nlp = spacy.load("en_core_web_sm")
# từ câu tách ra thành các danh từ và cụm danh từ
def Word_Processing(sentence):
  
  doc = nlp(sentence)
  filtered_words = [token.text for token in doc if not token.is_stop]

  filtered_sentence = ' '.join(filtered_words)
  
  nouns = []
  noun_phrases = []
  verbs = []

  for token in doc:
      if token.pos_ == "NOUN":
          nouns.append(token.text)
      if token.pos_ == "VERB":
          verbs.append(token.text)
  for chunk in doc.noun_chunks:
      noun_phrases.append(chunk.text)

  
  '''print("Nouns:", nouns)
  print("Noun Phrases:", noun_phrases)
  print("Verbs:", verbs)'''
  return noun_phrases
# Loại bỏ stop word
def pre_Word_Processing(sentence):

  nlp = spacy.load("en_core_web_sm")

  doc = nlp(sentence)

  filtered_words = [token.text for token in doc if not token.is_stop]

  filtered_sentence = ' '.join(filtered_words)
  #print("Filtered Sentence:", filtered_sentence)
  return filtered_sentence

def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a 2D matrix to store the edit distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]

# Vẽ đồ thị từ danh sách danh từ và cụm danh từ
def Make_Graph(noun_phrases):
  G = nx.Graph()

  for i in noun_phrases:
    G.add_node(i)
  for i in noun_phrases:
    G.add_node(i)
  for i in noun_phrases:
    for j in noun_phrases:
      if i<j :
        G.add_edge(i,j)

  return G

  pos = nx.spring_layout(G)  # Xác định vị trí của các node
  nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=4000, font_size=9)
  return G

sentence1= 'I have a dog and i love it'
sentence2= 'I have a cat and i hate it'

G1 = Make_Graph(Word_Processing(pre_Word_Processing(sentence1)))  
G2 = Make_Graph(Word_Processing(pre_Word_Processing(sentence2)))

# tính GED
def Caculate_GED(G1,G2):
  ged = nx.graph_edit_distance(G1,G2)
  return ged

ged=Caculate_GED(G1,G2)
print(1/(1+ged))