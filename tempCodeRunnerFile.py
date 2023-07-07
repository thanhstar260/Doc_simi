from fastapi import FastAPI, Form
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')