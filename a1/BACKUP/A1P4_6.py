import torch
import torch.nn as nn

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

        init_range = 0.5 / embedding_size
        self.embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, t):
        x_emb = self.embeddings(x)        
        t_emb = self.embeddings(t)       

        prediction = torch.sum(x_emb * t_emb, dim=1)  
        return prediction

if __name__ == "__main__":
    vocab_size = 1000
    embedding_size = 50
    model = SkipGramNegativeSampling(vocab_size, embedding_size)
    x = torch.tensor([1, 2, 3, 4])   
    t = torch.tensor([2, 3, 4, 5])   

    pred = model(x, t)
    print("Prediction shape:", pred.shape)   
    print("Raw prediction values:", pred)