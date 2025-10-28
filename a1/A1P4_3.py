import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import torch
import spacy

nltk.download('punkt')
nltk.download('punkt_tab')  # required for some sentence tokenization variants

def prepare_texts(text, min_frequency=3):
    nlp = spacy.load("en_core_web_sm")
    lemmas = []

    for sent in sent_tokenize(text):
        for tok in nlp(sent):
            # skip punctuation, spaces, numbers, symbols, or unknown tokens
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    freqs = Counter(lemmas)
    vocab = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    # keep only words that meet minimum frequency
    frequent_vocab = list(filter(lambda item: item[1] >= min_frequency, vocab))

    w2i = {word: idx for idx, (word, _) in enumerate(frequent_vocab)}
    i2w = {idx: word for word, idx in w2i.items()}

    # add <oov> token for rare words
    oov_idx = len(frequent_vocab)
    w2i["<oov>"] = oov_idx
    i2w[oov_idx] = "<oov>"

    # replace rare words with <oov>
    filtered_lemmas = [lem if lem in w2i else "<oov>" for lem in lemmas]

    return filtered_lemmas, w2i, i2w

# read the corpus
with open("LargerCorpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

filtered_lemmas, w2i, i2w = prepare_texts(text, min_frequency=3)

num_words = len(filtered_lemmas)
vocab_size = len(w2i) - 1  # exclude <oov> from count

freqs = Counter(filtered_lemmas)
top20 = freqs.most_common(20)

print("Number of words in the text:", num_words)
print("Size of filtered vocabulary:", vocab_size)
print("Top 20 most frequent words:")
for word, count in top20:
    print(f"{word}: {count}")
