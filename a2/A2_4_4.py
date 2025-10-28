# overfit_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Dataset class
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, file, word2idx):
        data = pd.read_csv(file, sep="\t")
        self.texts = data["text"].tolist()
        self.labels = data["label"].tolist()
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()
        indices = [self.word2idx.get(w, 0) for w in words]
        label = self.labels[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# ---------------------------
# Collate function (padding)
# ---------------------------
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(x) for x in texts]
    max_len = max(lengths)

    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, seq in enumerate(texts):
        padded[i, :len(seq)] = seq

    return padded.t(), torch.tensor(labels, dtype=torch.float)

# ---------------------------
# Baseline model (average embeddings)
# ---------------------------
class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)  # [seq_len, batch, embed_dim]
        emb = emb.mean(dim=0)    # average over sequence -> [batch, embed_dim]
        out = self.fc(emb).squeeze(1)
        return torch.sigmoid(out)

# ---------------------------
# Build vocab from overfit.tsv
# ---------------------------
def build_vocab(file):
    data = pd.read_csv(file, sep="\t")
    vocab = {"<PAD>": 0}
    idx = 1
    for text in data["text"]:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# ---------------------------
# Training loop
# ---------------------------
def train(model, dataloader, optimizer, criterion, epochs=30):
    train_losses, train_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for X, y in dataloader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    return train_losses, train_accs

# ---------------------------
# Main
# ---------------------------
def main():
    file = "overfit.tsv"  # Only use overfit dataset
    word2idx = build_vocab(file)

    dataset = TextDataset(file, word2idx)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = BaselineModel(vocab_size=len(word2idx), embed_dim=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    losses, accs = train(model, dataloader, optimizer, criterion, epochs=40)

    # Plot results
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.plot(accs, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Overfitting Debugging on overfit.tsv")
    plt.savefig("overfit_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
