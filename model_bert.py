import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class InCaseLawBERT(nn.Module):
    
    def __init__(self):
        super(InCaseLawBERT, self).__init__()
        self.model = AutoModel.from_pretrained("law-ai/InCaseLawBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("law-ai/InCaseLawBERT")
        self.relu = nn.ReLU()
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        encoded_input = self.tokenizer(x, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        encoded_input = encoded_input.to('cuda')
        output = self.model(**encoded_input)
        out = output.pooler_output
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
