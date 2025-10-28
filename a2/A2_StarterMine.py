import torch
import torch.optim as optim
import os
import pandas as pd
import urllib.request, zipfile
import numpy as np
from tqdm import tqdm

# ==============================================================
# Download + Load GloVe 100d vectors (global embeddings required)
# ==============================================================

def download_glove(glove_dir='./glove', glove_dim=100):
    """
    Download GloVe vectors (6B) of given dimension if not already present.
    """
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
    """
    Load GloVe vectors into word2idx, idx2word, and a torch FloatTensor.
    """
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


# actually load the 100-dim vectors
glove_path = download_glove(glove_dim=100)
word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

# Add UNK token (used for out-of-vocabulary words)
UNK_INDEX = len(word2idx)
word2idx["<UNK>"] = UNK_INDEX
idx2word.append("<UNK>")
embeddings = torch.cat([embeddings, embeddings[-1:].clone()], dim=0)  # copy last vector

def glove(word):
    """Return the GloVe vector for a given word (use <UNK> if missing)."""
    idx = word2idx.get(word, UNK_INDEX)
    return embeddings[idx]

# ==============================================================
# TextDataset is Described in Section 3.3 of Assignment 2
# ==============================================================


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        data_path = "./data"

        # print("path:", os.path.join(data_path, f"{split}.tsv"))

        df = pd.read_csv(os.path.join(data_path, f"{split}.tsv"), sep="\t")

        V = len(embeddings)

        # X: torch.tensor (maxlen, batch_size), padded indices; maxlen is computed below after all data put into X
        # Y: torch.tensor of length V
        X, Y = [], []

        for i, row in df.iterrows():
            L = row["text"].split()
            X.append(torch.tensor([word2idx.get(w, V-1) for w in L]))  # last vector is UNK
            Y.append(row.label)
        self.X = X
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# function to prepare batches and pad sequences in a batch to the same length
def my_collate_function(batch, device):
    # Handle the padding in this function
    batch_x, batch_y = [], []
    max_len = 0
    for x, y in batch:
        batch_y.append(y)
        max_len = max(max_len, len(x))
    for x, y in batch:
        x_p = torch.concat([x, torch.zeros(max_len - len(x))])
        batch_x.append(x_p)
    return torch.stack(batch_x).t().int().to(device), torch.tensor(batch_y).to(device)


def main(args):
    #   fix seed
    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # In this section of the code, you'll need to download the Glove Vectors as you did in Assignment 1
    # (Or presumably, if you've stored them locally, just load them)
    # The code above assumes that there is a global tensor called 'embeddings' that is these vectors available
    # Different from Assignment 1, you should use the embedding size 100, not 50.

    # 3.3.2
    train_dataset = TextDataset("train")
    val_dataset = TextDataset("validation")
    test_dataset = TextDataset("test")

    # 3.3.3
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    # Instantiate your model(s) and train them and so on
    # We suggest parameterizing the model - k1, n1, k2, n2, and other hyperparameters
    # so that it is easier to experiment with
