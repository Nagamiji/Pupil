import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden_states):
        energy = torch.tanh(self.attn(hidden_states))
        attn_scores = torch.einsum("bsh,h->bs", energy, self.v)
        return F.softmax(attn_scores, dim=1)

class HybridKhmerRecognizer(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, rnn_hidden_dim, num_layers, num_classes, dropout_prob=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.gru = nn.GRU(
            input_size=cnn_out_channels, hidden_size=rnn_hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, bidirectional=True
        )
        self.attention = Attention(rnn_hidden_dim * 2)
        self.fc1 = nn.Linear(rnn_hidden_dim * 2, rnn_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x).permute(0, 2, 1)
        gru_out, _ = self.gru(cnn_out)
        attn_weights = self.attention(gru_out)
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out)