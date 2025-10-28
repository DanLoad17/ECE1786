import os
import zipfile
import urllib.request
import numpy as np
import torch
from tqdm import tqdm


def download_glove(glove_dir="./glove", glove_dim=50):
    # Make sure the directory exists
    os.makedirs(glove_dir, exist_ok=True)

    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(glove_dir, "glove.6B.zip")

    # Download if not already present
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe vectors from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    glove_file = f"glove.6B.{glove_dim}d.txt"
    glove_path = os.path.join(glove_dir, glove_file)

    # Extract the specific file if missing
    if not os.path.exists(glove_path):
        print(f"Extracting {glove_file}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract(glove_file, glove_dir)

    return glove_path


def load_glove_vectors(glove_path, vocab_size=None):
    print(f"Loading GloVe vectors from {glove_path}...")

    word2idx, idx2word, vectors = {}, [], []

    with open(glove_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if vocab_size and i >= vocab_size:
                break
            parts = line.strip().split()
            word, vector = parts[0], np.array(parts[1:], dtype="float32")

            word2idx[word] = i
            idx2word.append(word)
            vectors.append(vector)

    embeddings = torch.from_numpy(np.array(vectors))
    print(f"Loaded {len(word2idx)} words with dimension {embeddings.shape[1]}")
    return word2idx, idx2word, embeddings


def glove(word):
    # Lookup helper for embeddings
    idx = word2idx.get(word)
    if idx is None:
        raise KeyError(f"Word '{word}' not found in GloVe vocabulary.")
    return embeddings[idx]


def cosine_similarity(vec1, vec2):
    # Normalized dot product
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))


def compare_word_to_category(word, category_words):
    word_vec = glove(word)
    category_vecs = [glove(w) for w in category_words]

    # Method a) -> average of individual similarities
    sims = [cosine_similarity(word_vec, cat_vec).item() for cat_vec in category_vecs]
    avg_similarity = sum(sims) / len(sims)

    # Method b) -> similarity with averaged category vector
    category_avg_vec = torch.mean(torch.stack(category_vecs), dim=0)
    category_avg_similarity = cosine_similarity(word_vec, category_avg_vec).item()

    return avg_similarity, category_avg_similarity


if __name__ == "__main__":
    glove_path = download_glove(glove_dim=50)
    word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

    temperature_category = [
        "temperature", "hot", "cold", "warm", "cool",
        "heat", "chill", "freeze", "boil"
    ]

    test_words = [
        "fire", "ice", "desert", "snow", "summer",
        "winter", "oven", "frozen", "lava", "computer"
    ]

    print(f"{'Word':<15} {'Method a)':<15} {'Method b)':<15}")
    print("--------------------------------------")

    for word in test_words:
        try:
            avg_sim, cat_avg_sim = compare_word_to_category(word, temperature_category)
            print(f"{word:<15} {avg_sim:<15.4f} {cat_avg_sim:<15.4f}")
        except KeyError:
            print(f"{word:<15} {'N/A':<15} {'N/A':<15}")
