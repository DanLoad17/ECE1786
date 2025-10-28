import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class Word2vecModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_size = 2 
        self.embeddings = nn.Embedding(vocab_size, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, vocab_size)

    def forward(self, x):
        e = self.embeddings(x)
        logits = self.linear(e)
        return logits, e

def prepare_texts(text):
    nlp = spacy.load("en_core_web_sm")
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    freqs = Counter(lemmas)
    vocab = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
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

def train_word2vec(textlist, window=5, embedding_size=2, lr=0.01, n_epochs=50):
    freqs = Counter(textlist)
    vocab_sorted = [w for w, _ in freqs.most_common()]
    v2i = {w: i for i, w in enumerate(vocab_sorted)}
    i2v = {i: w for w, i in v2i.items()}
    vocab_size = len(v2i)

    X_list, Y_list = tokenize_and_preprocess_text(textlist, v2i, window)
    X_tensor = torch.tensor(X_list, dtype=torch.long)
    Y_tensor = torch.tensor(Y_list, dtype=torch.long)
    dataset = TensorDataset(X_tensor, Y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    network = Word2vecModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        network.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = network(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_train = running_loss / train_size
        train_losses.append(avg_train)

        network.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                logits_val, _ = network(xb_val)
                loss_val = criterion(logits_val, yb_val)
                val_running += loss_val.item() * xb_val.size(0)
        avg_val = val_running / val_size
        val_losses.append(avg_val)

    return network, v2i, i2v, train_losses, val_losses

def plot_embeddings_2d(network, i2v, title="Word Embeddings 2D"):
    embeddings = network.embeddings.weight.detach().numpy()
    plt.figure(figsize=(8, 6))
    for i, word in i2v.items():
        x, y = embeddings[i]
        plt.scatter(x, y, color="blue")
        plt.text(x + 0.01, y + 0.01, word, fontsize=9)
    plt.title(title)
    plt.xlabel("Embedding D1")
    plt.ylabel("Embedding D2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    with open("SmallSimpleCorpus.txt", "r") as f:
        corpus_text = f.read()

    lemmas, v2i, i2v, vocab = prepare_texts(corpus_text)

    print("Training without fixed seeds")
    net1, _, _, _, _ = train_word2vec(lemmas) #Required help of AI to ask how to display multiple nets of lemmas within this context
    net2, _, _, _, _ = train_word2vec(lemmas)

    print("\nTraining with fixed seeds")
    np.random.seed(123)
    torch.manual_seed(123)
    net_fixed, _, _, _, _ = train_word2vec(lemmas)
    plot_embeddings_2d(net_fixed, i2v, title="Word Embeddings (Fixed Seed)")