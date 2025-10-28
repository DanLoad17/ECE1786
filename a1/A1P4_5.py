import random
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
from collections import Counter

def prepare_texts(text, min_frequency=3):
    nlp = spacy.load("en_core_web_sm")
    lemmas = []

    for sent in sent_tokenize(text):
        for tok in nlp(sent):
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    freqs = Counter(lemmas)
    vocab = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    frequent_vocab = [item for item in vocab if item[1] >= min_frequency]

    w2i = {w[0]: i for i, w in enumerate(frequent_vocab)}
    i2w = {i: w[0] for i, w in enumerate(frequent_vocab)}

    # add out-of-vocabulary token
    oov_idx = len(frequent_vocab)
    w2i["<oov>"] = oov_idx
    i2w[oov_idx] = "<oov>"

    filtered_lemmas = [w if w in w2i else "<oov>" for w in lemmas]
    return filtered_lemmas, w2i, i2w


def tokenize_and_preprocess_text_subsample(textlist, w2i, window, threshold=1e-5):
    X, T, Y = [], [], []

    # convert words to indices
    tokenized = [w2i.get(word, w2i["<oov>"]) for word in textlist]
    total_count = len(tokenized)

    # compute frequency of each token
    freqs = {}
    for tok in tokenized:
        freqs[tok] = freqs.get(tok, 0) + 1
    freqs = {w: c / total_count for w, c in freqs.items()}

    # subsampling to reduce very frequent words
    subsampled_tokens = []
    for tok in tokenized:
        f = freqs[tok]
        prob_keep = min(1.0, np.sqrt(threshold / f) + (threshold / f))
        if random.random() < prob_keep:
            subsampled_tokens.append(tok)

    print(f"Original tokens: {len(tokenized)}")
    print(f"Tokens after subsampling: {len(subsampled_tokens)}")

    vocab_size = len(w2i)

    for i, center in enumerate(subsampled_tokens):
        start = max(0, i - window)
        end = min(len(subsampled_tokens), i + window + 1)

        positive_samples = []
        for j in range(start, end):
            if j != i:
                context = subsampled_tokens[j]
                X.append(center)
                T.append(context)
                Y.append(1)  # positive pair
                positive_samples.append(context)

        # generate negative samples
        for _ in range(len(positive_samples)):
            neg_word = random.choice(subsampled_tokens)
            X.append(center)
            T.append(neg_word)
            Y.append(-1)  # negative pair

    return X, T, Y


if __name__ == "__main__":
    with open("LargerCorpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    filtered_lemmas, w2i, i2w = prepare_texts(text)
    X, T, Y = tokenize_and_preprocess_text_subsample(filtered_lemmas, w2i, window=5)

    print("Total examples created after subsampling:", len(Y))
    print("Positive examples:", Y.count(1))
    print("Negative examples:", Y.count(-1))
