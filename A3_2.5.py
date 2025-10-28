import torch
from torch.nn import functional as F

def generate(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    max_probs = torch.tensor([]).to(device)
    six_indices = torch.empty((0, 6), dtype=torch.int64, device=device)
    six_probs = torch.empty((0, 6), device=device)

    for _ in range(max_new_tokens):
        if idx.size(1) > self.block_size:
            idx_cond = idx[:, -self.block_size:]
        else:
            idx_cond = idx
        
        idx_cond = idx_cond.to(device)
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
            prob_next = probs.gather(1, idx_next)
        else:
            prob_next, idx_next = torch.topk(probs, k=6, dim=-1)

        next_token = idx_next[0][0].reshape(1, 1)
        idx = torch.cat((idx, next_token), dim=1)

        max_probs = torch.cat((max_probs, prob_next[0][0].reshape(1, 1)), dim=0)
        six_indices = torch.cat((six_indices, idx_next), dim=0)
        six_probs = torch.cat((six_probs, prob_next), dim=0)

    return idx, max_probs, six_indices, six_probs
