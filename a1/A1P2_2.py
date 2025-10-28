import os
import zipfile
import urllib.request

import numpy as np
import torch
from tqdm import tqdm


def download_glove(target_dir="./glove", dim=50):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "glove.6B.zip")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    if not os.path.exists(zip_path):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(url, zip_path)

    glove_txt = f"glove.6B.{dim}d.txt"
    glove_file = os.path.join(target_dir, glove_txt)

    if not os.path.exists(glove_file):
        print(f"Extracting {glove_txt}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(glove_txt, target_dir)

    return glove_file


def load_glove(path, vocab_size=None):
    print(f"Loading embeddings from {path}...")
    word_to_idx, idx_to_word, vectors = {}, [], []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if vocab_size and i >= vocab_size:
                break
            parts = line.rstrip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype="float32")
            word_to_idx[word] = i
            idx_to_word.append(word)
            vectors.append(vec)

    emb = torch.tensor(np.array(vectors))
    print(f"Loaded {len(word_to_idx)} words, dimension {emb.shape[1]}")
    return word_to_idx, idx_to_word, emb


def get_vector(word):
    idx = word_to_idx.get(word)
    if idx is None:
        raise KeyError(f"'{word}' not in vocabulary")
    return embeddings[idx]


def cosine_sim(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def compare_to_category(word, category_words):
    w_vec = get_vector(word)
    cat_vecs = [get_vector(w) for w in category_words]

    # method a -> average similarity across all category words
    sims = [cosine_sim(w_vec, v).item() for v in cat_vecs]
    method_a = sum(sims) / len(sims)

    # method b -> similarity with the average category vector
    avg_vec = torch.mean(torch.stack(cat_vecs), dim=0)
    method_b = cosine_sim(w_vec, avg_vec).item()

    return method_a, method_b


if __name__ == "__main__":
    glove_file = download_glove(dim=50)
    word_to_idx, idx_to_word, embeddings = load_glove(glove_file)

    colour_words = ["colour", "red", "green", "blue", "yellow"]

    test_words = [
        "greenhouse", "sky", "grass", "redwood", "sunshine",
        "microphone", "president", "coal", "lemon", "ember", "snow"
    ]

    print(f"{'Word':<15} {'Method a)':<15} {'Method b)':<15}")
    print("--------------------------------------")

    for w in test_words:
        try:
            m_a, m_b = compare_to_category(w, colour_words)
            print(f"{w:<15} {m_a:<15.4f} {m_b:<15.4f}")
        except KeyError:
            print(f"{w:<15} {'N/A':<15} {'N/A':<15}")
