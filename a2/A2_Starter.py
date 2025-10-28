
import torch
import torch.optim as optim
import os
import pandas as pd

# TextDataset is Described in Section 3.3 of Assignment 2

# The word 'embeddings' the same as Assignment #1 must be available as a global 


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
            X.append(torch.tensor([word2idx.get(w, V-1) for w in L]))  # I'm using the last word in the vocab as the "out-of-vocabulary" token
            
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
    # batch is approximately: [dataset[i] for i in range(0, batch_size)]
    # Since dataset[i]'s contents is defined in the __getitem__() above, this collate function should be set correspondingly.
    # Also: collate_function just takes one argument. To pass in additional arguments (e.g., device), you need to wrap up an anonymous function (using lambda below)
    batch_x, batch_y = [], []
    max_len = 0
    for x,y in batch:
        batch_y.append(y)
        max_len = max(max_len, len(x))
    for x,y in batch:
        x_p = torch.concat(
            [x, torch.zeros(max_len - len(x))]
        )
        batch_x.append(x_p)
    return torch.stack(batch_x).t().int().to(device), torch.tensor(batch_y).to(device)


def main(args):
    #   fix seed
    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)


# In this section of the code, you'll need to download the Glove Vectors as you did in Assignment 1
# (Or presumably, if you've stored them locally, just load them)
# The code above assumes that there is a global tensor called 'embeddings' that is these vectors available
# Different from Assignment 1, you should use the embedding size 100, not 50.

# Your Code goes here

                                   
    # 3.3.2

    train_dataset = TextDataset(glove, "train")
    val_dataset = TextDataset(glove, "validation")
    test_dataset = TextDataset(glove, "test")
        
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
    
    
   
