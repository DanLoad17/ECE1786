import torch
import numpy as np
import zipfile
import urllib.request
import os
from tqdm import tqdm

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

def print_closest_words(word, n=5):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    vec = embeddings[word2idx[word]]
    dists = torch.norm(embeddings - vec, dim=1)  
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])  

    results = []
    for idx, difference in lst[1:n+1]: 
        results.append((idx2word[idx], difference))
    return results

def print_closest_cosine_words(word, N=5):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    word_vec = embeddings[word2idx[word]].unsqueeze(0)
    normed_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    normed_word_vec = torch.nn.functional.normalize(word_vec, p=2, dim=1)

    similarities = torch.mm(normed_embeddings, normed_word_vec.T).squeeze()
    sorted_sim, sorted_idx = torch.sort(similarities, descending=True)

    results = []
    for i in range(1, N+1):  
        sim_word = idx2word[sorted_idx[i].item()]
        results.append((sim_word, sorted_sim[i].item()))
    return results

if __name__ == "__main__":
    glove_path = download_glove(glove_dim=50)
    word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

    test_words = ["dog", "computer"]

    for word in test_words:
        cos_results = print_closest_cosine_words(word, N=5)
        euc_results = print_closest_words(word, n=10)

        print(f"\n")
        print(f"Finding closest words to: {word}")
        print(f"{'Cosine Similarity':35s} | {'Euclidean Distance':35s}")
        print("---------------------------------------")

        for i in range(max(len(cos_results), len(euc_results))):
            cos_str = f"{cos_results[i][0]} ({cos_results[i][1]:.4f})" if i < len(cos_results) else ""
            euc_str = f"{euc_results[i][0]} ({euc_results[i][1]:.2f})" if i < len(euc_results) else ""
            print(f"{cos_str:35s} | {euc_str:35s}")