from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split

def prepare_texts(text):    
    nlp = spacy.load("en_core_web_sm")
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    
    freqs = Counter() 
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True) 

    v2i = {v[0]:i for i,v in enumerate(vocab)}
    i2v = {i:v[0] for i,v in enumerate(vocab)}
    
    return lemmas, v2i, i2v, vocab

with open("SmallSimpleCorpus.txt", "r") as f:
    corpus_text = f.read()

lemmas, v2i, i2v, vocab = prepare_texts(corpus_text)

print("Vocabulary size:", len(vocab))   

most_freq = vocab[0]
least_freq = vocab[-1]
print("Most frequent word:", most_freq)
print("Least frequent word:", least_freq)
print("v2i (word, index):", v2i)
print("i2v (index, word):", i2v)