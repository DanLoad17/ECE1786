from collections import Counter
import torch
import spacy
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_texts(text):
    nlp = spacy.load("en_core_web_sm")
    # tokenize and lemmatize, ignore punctuation and spaces
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]

    freqs = Counter(lemmas)  # count word frequencies

    vocab = sorted(freqs.items(), key=lambda item: item[1], reverse=True)  # sort by frequency

    v2i = {word: i for i, (word, _) in enumerate(vocab)}  # word -> index
    i2v = {i: word for i, (word, _) in enumerate(vocab)}  # index -> word

    return lemmas, v2i, i2v, vocab

# read corpus
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
