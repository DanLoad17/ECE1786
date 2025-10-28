import torch

class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_size = 2  
        
        self.embeddings = torch.nn.Embedding(vocab_size, self.embedding_size)
        
        self.linear = torch.nn.Linear(self.embedding_size, vocab_size)
        
    def forward(self, x):
        e = self.embeddings(x)        
        logits = self.linear(e)      
        return logits, e