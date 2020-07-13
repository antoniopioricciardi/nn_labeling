import torch.nn as nn
import torch.nn.functional as F

'''
Model is composed of the EmbeddingBag layer and the linear layer.
nn.EmbeddingBag computer the mean value of a "bag" of embeddings'''

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # nn.EmbeddingBag.from_pretrained()
        # nn.Embedding.from_pretrained(freeze=True)
        self.embedding.weight.requires_grad = False
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)