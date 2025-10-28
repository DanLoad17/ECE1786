import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

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
            Y.append(0)  

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

def train_sgns(textlist, w2i, window=5, embedding_size=8, epochs=30):
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    data = torch.tensor(list(zip(X, T)), dtype=torch.long)
    labels = torch.tensor(Y, dtype=torch.float)

    dataset = TensorDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    vocab_size = len(w2i)
    model = SkipGramNegativeSampling(vocab_size, embedding_size)
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    eps = 1e-5
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_data, batch_labels in train_loader:
            word, context = batch_data[:, 0], batch_data[:, 1]
            preds = model(word, context)

            loss = -torch.mean(
                batch_labels * torch.log(torch.sigmoid(preds) + eps) +
                (1 - batch_labels) * torch.log(torch.sigmoid(-preds) + eps))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                word, context = batch_data[:, 0], batch_data[:, 1]
                preds = model(word, context)
                loss = -torch.mean(
                    batch_labels * torch.log(torch.sigmoid(preds) + eps) +
                    (1 - batch_labels) * torch.log(torch.sigmoid(-preds) + eps)
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SGNS Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png")
    plt.show()

    return model

#Required help of AI to perform PCA in the context of i2w
def visualize_embedding(embedding, i2w, freqs, w2i, most_frequent_from=20, most_frequent_to=80):
    print(f"Visualizing words ranked {most_frequent_from} to {most_frequent_to} by frequency")

    ordered_words = [w for w, _ in freqs.most_common()]
    selected_words = ordered_words[most_frequent_from:most_frequent_to]

    selected_indices = [w2i[w] for w in selected_words if w in w2i]
    selected_embeddings = embedding[selected_indices, :]

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(selected_embeddings)

    plt.figure(figsize=(10, 7))
    for i, coord in enumerate(embeddings_2d):
        word = selected_words[i]
        plt.scatter(coord[0], coord[1], marker="o", color="blue")
        plt.text(coord[0] + 0.02, coord[1] + 0.02, word, fontsize=9)

    plt.title(f"PCA of word embeddings ({most_frequent_from}-{most_frequent_to} most frequent)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("pca_embeddings.png")
    plt.show()

if __name__ == "__main__":
    corpus_path = "LargerCorpus.txt" 
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read().lower().split()

    freqs = Counter(raw_text)
    vocab = [w for w, _ in freqs.most_common()]
    vocab.append("<oov>")
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}

    print(f"Vocabulary size: {len(vocab)}")

    model = train_sgns(raw_text, w2i, window=5, embedding_size=8, epochs=30)
    embedding_matrix = model.embeddings.weight.detach().numpy()

    visualize_embedding(embedding_matrix, i2w, freqs, w2i,
                        most_frequent_from=20, most_frequent_to=80)