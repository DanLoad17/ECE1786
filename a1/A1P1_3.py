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

    # download zip if not already there
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe vectors from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    glove_file = f'glove.6B.{glove_dim}d.txt'
    glove_path = os.path.join(glove_dir, glove_file)

    # extract the specific glove file if not already extracted
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

    # read glove file line by line
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

    # turn vectors into torch tensor
    embeddings = torch.from_numpy(np.array(vectors))
    print(f"Loaded {len(word2idx)} words with dimension {embeddings.shape[1]}")
    return word2idx, idx2word, embeddings

def glove(word):
    # return embedding for word if it exists
    idx = word2idx.get(word)
    if idx is not None:
        return embeddings[idx]
    else:
        raise KeyError(f"Word '{word}' not found in GloVe vocabulary.")

def print_closest_words(vec, n=5):
    # euclidean distance to all words
    dists = torch.norm(embeddings - vec, dim=1)    
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) 
    # skip first (same word) and print top n
    for idx, difference in lst[1:n+1]:             
        print(idx2word[idx], "\t%5.2f" % difference)

if __name__ == "__main__":
    glove_path = download_glove(glove_dim=50)
    word2idx, idx2word, embeddings = load_glove_vectors(glove_path)

    print("\nAnalogy: Animal to Baby version")

    animals = ["cat", "dog", "cow", "horse", "sheep"]

    for animal in animals:
        print(f"\nAnimal: {animal}")
        try:
            # analogy: animal - "animal" + "baby"
            vec = glove(animal) - glove("animal") + glove("baby")
            print_closest_words(vec, n=5)
        except KeyError:
            print(f"Word '{animal}' not in GloVe vocabulary.")
