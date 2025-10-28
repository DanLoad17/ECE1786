import os
import sys
import subprocess
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

def tokenize_and_preprocess_text(textlist, w2i, window):
    X, T, Y = [], [], []
    tokenized = [w2i.get(word, w2i["<oov>"]) for word in textlist]
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

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x, t):
        embed_x = self.embeddings(x)
        embed_t = self.embeddings(t)
        prediction = torch.sum(embed_x * embed_t, dim=1)
        return prediction

def open_image_with_default_viewer(filepath):
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", filepath])
        elif os.name == "nt":
            os.startfile(filepath)  
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", filepath])
        else:
            print(f"Cannot auto-open file on this platform. File saved at: {filepath}")
    except Exception as e:
        print("Failed to open image automatically:", e)
        print("Open the saved file manually:", filepath)

def train_sgns(textlist, w2i, window=5, embedding_size=8):
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    data = torch.tensor(list(zip(X, T)), dtype=torch.long)
    labels = torch.tensor([1 if y == 1 else 0 for y in Y], dtype=torch.float)
    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    vocab_size = len(w2i)
    model = SkipGramNegativeSampling(vocab_size, embedding_size)
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 30
    eps = 1e-5
    train_losses, val_losses = [], []

    torch.manual_seed(0)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        for batch_data, batch_labels in train_loader:
            word, context = batch_data[:,0], batch_data[:,1]
            optimizer.zero_grad()
            preds = model(word, context)
            loss = -torch.mean(
                batch_labels * torch.log(torch.sigmoid(preds)+eps) +
                (1 - batch_labels) * torch.log(torch.sigmoid(-preds)+eps)
            )
            loss.backward()
            optimizer.step()
            running_train += loss.item() * word.size(0)
        avg_train = running_train / (train_size if train_size > 0 else 1)
        train_losses.append(avg_train)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                word, context = batch_data[:,0], batch_data[:,1]
                preds = model(word, context)
                loss = -torch.mean(
                    batch_labels * torch.log(torch.sigmoid(preds)+eps) +
                    (1 - batch_labels) * torch.log(torch.sigmoid(-preds)+eps)
                )
                running_val += loss.item() * word.size(0)
        avg_val = running_val / (val_size if val_size > 0 else 1)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {avg_train:.6f} - Val Loss: {avg_val:.6f}")

    out_path = os.path.abspath("sgns_loss_curve.png")
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SGNS Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss curve to: {out_path}")
    open_image_with_default_viewer(out_path)
    print(f"Learning rate: {learning_rate}")
    return model

if __name__ == "__main__":
    corpus_path = "SmallSimpleCorpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: corpus file not found at {corpus_path}. Place SmallSimpleCorpus.txt in this folder.")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    words = raw_text.lower().split()
    freqs = {w: words.count(w) for w in set(words)}
    sorted_words = sorted(freqs, key=freqs.get, reverse=True)
    v2i = {w:i for i,w in enumerate(sorted_words)}
    if "<oov>" not in v2i:
        v2i["<oov>"] = len(v2i)

    model = train_sgns(words, v2i, window=5, embedding_size=8)
    print("\nTraining finished.")