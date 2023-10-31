import torch
import torch.nn as nn

class BiLSMTNet(nn.Module):
    
    def __init__(self, input_size):
        
        super(BiLSMTNet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(input_size, 32, padding="same", kernel_size=3)
        self.pooling = nn.MaxPool1d(kernel_size=1)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, bidirectional=True, batch_first=False)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        x = x['input_ids'].to('cuda')
        x = torch.permute(x, (0, 2, 1))
        x = x.float()
        out = self.relu(self.conv1d(x))
        out = self.pooling(out)
        out = torch.permute(out, (0, 2, 1))
        out = torch.permute(out, (1, 0, 2))
        out,_ = self.lstm1(out)
        out = self.dropout(out[-1])
        out,_ = self.lstm2(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
