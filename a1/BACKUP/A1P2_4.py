import torch
import numpy as np
import zipfile
import urllib.request
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def download_glove(glove_dir='./glove', glove_dim=50):
    os.makedirs(glove_dir, exist_ok=True)
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(glove_dir, 'glove.6B.zip')
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe vectors from {url}...")
        urllib.request.urlretrieve(url, zip_path)
    glove_file = f'glove.6B.{glove_dim}d.txt'
    glove_path = os.path.join(glove_dir, glove_file)
    if not os.path.exists(glove_path):
        print(f"Extracting {glove_file}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(glove_file, glove_dir)
    return glove_path

def load_glove_vectors(glove_path, vocab_size=None):
    print(f"Loading GloVe vectors from {glove_path}...")
    word2idx = {}
    idx2word = []
    vectors = []
    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if vocab_size and i >= vocab_size:
                break
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word2idx[word] = i
            idx2word.append(word)
            vectors.append(vector)
    embeddings = torch.from_numpy(np.array(vectors))
    print(f"Loaded {len(word2idx)} words with dimension {embeddings.shape[1]}")
    return word2idx, idx2word, embeddings

def glove(word):
    idx = word2idx.get(word)
    if idx is not None:
        return embeddings[idx]
    else:
        raise KeyError(f"Word '{word}' not found in GloVe vocabulary.")

def cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

def compare_word_to_category(word, category_words):
    word_vec = glove(word)
    category_vecs = [glove(w) for w in category_words]
    category_avg_vec = torch.mean(torch.stack(category_vecs), dim=0)
    return cosine_similarity(word_vec, category_avg_vec).item()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    glove_path = download_glove(glove_dim=50)
    word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

    colour_category = ["colour", "red", "green", "blue", "yellow"]
    temperature_category = ["temperature", "hot", "cold", "warm", "cool", "heat", "chill", "freeze", "boil"]

    test_words = ["sun", "moon", "winter", "rain", "cow", "wrist",
                  "wind", "prefix", "ghost", "glow", "heated", "cool"]

    vectors_2d = {}
    for word in test_words:
        try:
            colour_sim = compare_word_to_category(word, colour_category)
            temp_sim = compare_word_to_category(word, temperature_category)

            vec = np.array([colour_sim, temp_sim])
            vec_prob = softmax(vec)

            vectors_2d[word] = vec_prob
        except KeyError:
            print(f"Word '{word}' not found in vocabulary.")

    plt.figure(figsize=(8, 6))
    for word, vec in vectors_2d.items():
        plt.scatter(vec[0], vec[1], label=word)
        plt.text(vec[0]+0.01, vec[1]+0.01, word, fontsize=9)

    plt.xlabel("Colour similarity")
    plt.ylabel("Temperature similarity")
    plt.title("Colour vs Temperature")
    plt.grid(True)
    plt.show()