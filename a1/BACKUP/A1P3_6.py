import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
import spacy

def prepare_texts(text):    
    nlp = spacy.load("en_core_web_sm")
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    freqs = Counter(lemmas) 
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

def train_word2vec(textlist, window, embedding_size):
    if window < 3 or window % 2 == 0:
        raise ValueError("Window must be odd integer >=3")
    if embedding_size != 2:
        raise ValueError("Embedding size must be 2")

    freqs = Counter(textlist)
    vocab_sorted = [w for w, _ in freqs.most_common()]
    v2i = {w: i for i, w in enumerate(vocab_sorted)}
    i2v = {i: w for w, i in v2i.items()}
    vocab_size = len(v2i)

    X_list, Y_list = tokenize_and_preprocess_text(textlist, v2i, window)
    X_tensor = torch.tensor(X_list, dtype=torch.long)
    Y_tensor = torch.tensor(Y_list, dtype=torch.long)
    dataset = TensorDataset(X_tensor, Y_tensor)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    network = Word2vecModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = optim.Adam(network.parameters(), lr=lr)
    print(f"Adam optimizer with lr = {lr}")

    n_epochs = 50
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

        print(f"Epoch {epoch:02d}/{n_epochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, n_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return network, v2i, i2v

def plot_embeddings_2d(network, i2v):
    embeddings = network.embeddings.weight.data.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color="blue")
    for i, word in i2v.items():
        plt.text(embeddings[i, 0]+0.01, embeddings[i, 1]+0.01, word, fontsize=9)
    plt.title("2D Word Embeddings")
    plt.xlabel("Embedding D1")
    plt.ylabel("Embedding D2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    with open("SmallSimpleCorpus.txt", "r") as f:
        corpus_text = f.read()

    lemmas, v2i, i2v, vocab = prepare_texts(corpus_text)
    network, v2i, i2v = train_word2vec(lemmas, window=5, embedding_size=2)
    plot_embeddings_2d(network, i2v)