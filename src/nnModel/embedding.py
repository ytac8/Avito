import torch.nn as nn


class Embed(nn.Module):

    def __init__(self, num_emb, emb_dim):
        self.embedding = nn.Embedding(num_emb, emb_dim)

    def forward(self, input):
        output = self.embedding(input)
        return output
