import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class Pretrained_Embedding(nn.Module):
    def __init__(self, config):
        super(Pretrained_Embedding, self).__init__()

        self.pretrained_name = config['text_embedding']['pretrained_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.embedding = AutoModel.from_pretrained(self.pretrained_name)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.max_length = config['text_embedding']['max_length']
        self.truncation = config['text_embedding']['truncation']

    def forward(self, text):
        inputs = self.tokenizer(text= text,
                                padding= 'max_length',
                                max_length= self.max_length,
                                truncation= self.truncation,
                                return_tensors= 'pt',
                                return_attention_mask= True).to(self.device)
        
        output = self.embedding(**inputs)

        return output
        
