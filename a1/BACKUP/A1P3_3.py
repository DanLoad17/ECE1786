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
    print("Vocabulary (word, count):", vocab)
    
    v2i = {v[0]: i for i, v in enumerate(vocab)}
    i2v = {i: v[0] for i, v in enumerate(vocab)}
    
    return lemmas, v2i, i2v, vocab

def tokenize_and_preprocess_text(textlist, v2i, window):
    X, Y = [], []  
    half_window = window // 2  
    for i, word in enumerate(textlist):
        target_idx = v2i[word] 
        
        for j in range(max(0, i - half_window), min(len(textlist), i + half_window + 1)):
            if j != i:  
                context_idx = v2i[textlist[j]]
                X.append(target_idx)
                Y.append(context_idx)
    
    return X, Y

if __name__ == "__main__":
    with open("SmallSimpleCorpus.txt", "r") as f:
        corpus_text = f.read()
    
    lemmas, v2i, i2v, vocab = prepare_texts(corpus_text)
    
    print("\nVocabulary size:", len(vocab))
    print("Most frequent word:", vocab[0])
    print("Least frequent word:", vocab[-1])
    
    window = 3
    X, Y = tokenize_and_preprocess_text(lemmas, v2i, window)
    
    print("\nNumber of pairs generated:", len(X))
    print("First 50:")
    for x, y in list(zip(X, Y))[:50]:
        print(f"{i2v[x]}, {i2v[y]}")