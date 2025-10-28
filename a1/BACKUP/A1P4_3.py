import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import time
import torch
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import nltk
nltk.download('punkt')
nltk.download("punkt_tab")
import spacy
from collections import Counter

def prepare_texts(text, min_frequency=3):
    nlp = spacy.load("en_core_web_sm")
    lemmas = []
    for sent in sent_tokenize(text):  
        for tok in nlp(sent):         
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)
    
    freqs = Counter()  
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  
    frequent_vocab = list(filter(lambda item: item[1] >= min_frequency, vocab))
    
    w2i = {w[0]: i for i, w in enumerate(frequent_vocab)}
    i2w = {i: w[0] for i, w in enumerate(frequent_vocab)}
    
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"
    
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)
    
    return filtered_lemmas, w2i, i2w

with open("LargerCorpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

filtered_lemmas, w2i, i2w = prepare_texts(text, min_frequency=3)

num_words = len(filtered_lemmas)

vocab_size = len(w2i) - 1  

freqs = Counter(filtered_lemmas)
top20 = freqs.most_common(20)

print("Number of words in the text:", num_words)
print("Size of filtered vocabulary:", vocab_size)
print("Top 20 most frequent words:")
for word, count in top20:
    print(f"{word}: {count}")