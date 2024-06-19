import torch
from torch import nn
from embedding.pretrained_embedding import Pretrained_Embedding


class Classifier_Model(nn.Module):
    def __init__(self, config):
        super(Classifier_Model, self).__init__()
        self.embedding = Pretrained_Embedding(config)
        self.classifier = nn.Linear(self.embedding.embedding.config.hidden_size, config["model"]["num_labels"])

    def forward(self, texts):
        embedding = self.embedding(texts)
        cls_embedding = embedding.last_hidden_state[:,0,:]
        logits = self.classifier(cls_embedding)

        return logits