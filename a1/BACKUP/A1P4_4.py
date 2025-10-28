import numpy as np
import random

def tokenize_and_preprocess_text(textlist, w2i, window):
    X, T, Y = [], [], []

    tokenized = [w2i.get(word, w2i["<oov>"]) for word in textlist]
    vocab_size = len(w2i)

    for i, center in enumerate(tokenized):
        start = max(0, i - window)
        end = min(len(tokenized), i + window + 1)

        positive_samples = []
        for j in range(start, end):
            if j != i:
                context = tokenized[j]
                X.append(center)
                T.append(context)
                Y.append(1)
                positive_samples.append(context)

        for _ in range(len(positive_samples)):
            neg_word = random.choice(tokenized)
            X.append(center)
            T.append(neg_word)
            Y.append(-1)

    return X, T, Y

if __name__ == "__main__":
    from collections import Counter
    import spacy
    from nltk.tokenize import sent_tokenize

    with open("LargerCorpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

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
        w2i["<oov>"] = len(frequent_vocab)
        i2w[len(frequent_vocab)] = "<oov>"

        filtered_lemmas = [w if w in w2i else "<oov>" for w in lemmas]
        return filtered_lemmas, w2i, i2w

    filtered_lemmas, w2i, i2w = prepare_texts(text)
    X, T, Y = tokenize_and_preprocess_text(filtered_lemmas, w2i, window=5)

    print("Total examples created:", len(Y))
    print("Positive examples:", Y.count(1))
    print("Negative examples:", Y.count(-1))