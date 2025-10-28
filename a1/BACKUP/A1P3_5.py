import os
import sys
import subprocess
import time
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import spacy  

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

def tokenize_and_preprocess_text(textlist, v2i, window):
    X, Y = [], []
    half_window = window // 2
    n = len(textlist)
    for i, word in enumerate(textlist):
        if word not in v2i:
            raise KeyError(f"Word '{word}' not in v2i")
        target_idx = v2i[word]
        left = max(0, i - half_window)
        right = min(n - 1, i + half_window)
        for j in range(left, right + 1):
            if j == i:
                continue
            context_word = textlist[j]
            if context_word not in v2i:
                raise KeyError(f"Word '{context_word}' not in v2i")
            context_idx = v2i[context_word]
            X.append(target_idx)
            Y.append(context_idx)
    return X, Y

def prepare_texts(text):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "  python -m pip install -U spacy\n"
            "  python -m spacy download en_core_web_sm\n"
        ) from e

    doc = nlp(text)
    lemmas = [tok.lemma_.lower() for tok in doc if tok.pos_ not in ("PUNCT", "SPACE")]
    freqs = Counter(lemmas)
    vocab_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True) 
    vocab_words = [w for w, _ in vocab_freqs]
    v2i = {w: i for i, w in enumerate(vocab_words)}
    i2v = {i: w for i, w in enumerate(vocab_words)}
    print("Vocabulary (word, count):", vocab_freqs)
    return lemmas, v2i, i2v, vocab_freqs

def open_image_with_default_viewer(filepath):
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", filepath])
        elif os.name == "nt":
            os.startfile(filepath)  
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", filepath])
        else:
            print(f"Can't auto-open file on this platform ({sys.platform}). File saved at: {filepath}")
    except Exception as e:
        print("Failed to open image automatically:", e)
        print("Open the saved file manually:", filepath)

def train_word2vec(textlist, window=5, embedding_size=2):
    if window < 3 or (window % 2) == 0:
        raise ValueError("window must be odd integer >= 3")
    if embedding_size != 2:
        raise ValueError("embedding_size must be 2 for this assignment.")

    freqs = Counter(textlist)
    vocab_words = [w for w, _ in freqs.most_common()]
    v2i = {w: i for i, w in enumerate(vocab_words)}
    i2v = {i: w for w, i in v2i.items()}
    vocab_size = len(vocab_words)
    print(f"Vocab size = {vocab_size}. Top words: {vocab_words[:10]}")

    X_list, Y_list = tokenize_and_preprocess_text(textlist, v2i, window)
    if len(X_list) == 0:
        raise RuntimeError("No pairs generated.")

    X_tensor = torch.tensor(X_list, dtype=torch.long)
    Y_tensor = torch.tensor(Y_list, dtype=torch.long)
    dataset = TensorDataset(X_tensor, Y_tensor)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Word2vecModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Adam optimizer with learning rate = {lr}")

    n_epochs = 50
    train_losses = []
    val_losses = []
    torch.manual_seed(0)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            xb = xb.view(-1)
            yb = yb.view(-1)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * xb.size(0)
        avg_train = running_train / (train_size if train_size > 0 else 1)
        train_losses.append(avg_train)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.view(-1)
                yb = yb.view(-1)
                logits_val, _ = model(xb)
                loss_val = criterion(logits_val, yb)
                running_val += loss_val.item() * xb.size(0)
        avg_val = running_val / (val_size if val_size > 0 else 1)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:02d}/{n_epochs} - Train Loss: {avg_train:.6f} - Val Loss: {avg_val:.6f}")

    out_path = os.path.abspath("loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, n_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved image of curve to: {out_path}")

    open_image_with_default_viewer(out_path)

    return model, v2i, i2v

if __name__ == "__main__":
    corpus_path = "SmallSimpleCorpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: corpus file not found in path {corpus_path}.")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    try:
        lemmas, v2i_prep, i2v_prep, vocab_freqs = prepare_texts(raw_text)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    model, v2i, i2v = train_word2vec(lemmas, window=5, embedding_size=2)