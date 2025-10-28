import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import urllib.request, zipfile
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# ============================================================== #
# Download + Load GloVe 100d vectors
# ============================================================== #
def download_glove(glove_dir='./glove', glove_dim=100):
    os.makedirs(glove_dir, exist_ok=True)
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    zip_path = os.path.join(glove_dir, 'glove.6B.zip')

    if not os.path.exists(zip_path):
        print(f"Downloading GloVe {glove_dim}d vectors ...")
        urllib.request.urlretrieve(url, zip_path)

    glove_file = f'glove.6B.{glove_dim}d.txt'
    glove_path = os.path.join(glove_dir, glove_file)

    if not os.path.exists(glove_path):
        print(f"Extracting {glove_file} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(glove_file, glove_dir)

    return glove_path


def load_glove_vectors(glove_path, vocab_size=None):
    print(f"Loading GloVe vectors from {glove_path} ...")
    word2idx, idx2word, vectors = {}, [], []

    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if vocab_size and i >= vocab_size:
                break
            vals = line.strip().split()
            word = vals[0]
            vec = np.array(vals[1:], dtype='float32')
            word2idx[word] = i
            idx2word.append(word)
            vectors.append(vec)

    embeddings = torch.from_numpy(np.stack(vectors))
    print(f"Loaded {len(word2idx)} words | dim={embeddings.shape[1]}")
    return word2idx, idx2word, embeddings


# Load GloVe embeddings
glove_path = download_glove(glove_dim=100)
word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

# Add UNK token
UNK_INDEX = len(word2idx)
word2idx["<UNK>"] = UNK_INDEX
idx2word.append("<UNK>")
embeddings = torch.cat([embeddings, embeddings[-1:].clone()], dim=0)


# ============================================================== #
# Dataset
# ============================================================== #
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        data_path = "./data"
        df = pd.read_csv(os.path.join(data_path, f"{split}.tsv"), sep="\t")
        V = len(embeddings)

        X, Y = [], []
        for _, row in df.iterrows():
            L = row["text"].split()
            X.append(torch.tensor([word2idx.get(w, V-1) for w in L]))
            Y.append(row.label)
        self.X = X
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def my_collate_function(batch, device):
    batch_x, batch_y = [], []
    max_len = 0
    for x, y in batch:
        batch_y.append(y)
        max_len = max(max_len, len(x))
    for x, y in batch:
        x_p = torch.concat([x, torch.zeros(max_len - len(x))])
        batch_x.append(x_p)
    return torch.stack(batch_x).t().int().to(device), torch.tensor(batch_y).to(device)


# ============================================================== #
# Baseline Model
# ============================================================== #
class BaselineModel(nn.Module):
    def __init__(self, embeddings, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
        emb_dim = embeddings.shape[1]
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)               # [seq_len, batch_size, emb_dim]
        sent_emb = emb.mean(dim=0)            # [batch_size, emb_dim]
        logits = self.fc(sent_emb).squeeze(1) # [batch_size]
        return logits


# ============================================================== #
# Evaluation Function
# ============================================================== #
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            loss = criterion(logits, y.float())
            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


# ============================================================== #
# Main Training Loop with Saving Best Model
# ============================================================== #
def main(args):
    torch.manual_seed(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Data
    train_dataset = TextDataset("train")
    val_dataset = TextDataset("validation")
    test_dataset = TextDataset("test")

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: my_collate_function(batch, device))

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    # Model
    model = BaselineModel(embeddings, freeze=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        val_loss, val_acc = evaluate(model, val_dataloader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model_baseline.pt")
            print("Saved new best model with val_loss:", val_loss)

    # Plot curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curves")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curves")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png")

    # Load best model before testing
    best_model = BaselineModel(embeddings, freeze=True).to(device)
    best_model.load_state_dict(torch.load("model_baseline.pt"))
    best_model.eval()

    # Final Test
    test_loss, test_acc = evaluate(best_model, test_dataloader, criterion)
    print(f"Final Test Accuracy (best model): {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="mini-batch size")
    args = parser.parse_args()
    main(args)
