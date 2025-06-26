import torch
import torch.nn as nn

class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_output_size, correctness_output_size, dropout_rate):
        super(DeepLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.shared_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.shared_fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.label_head = nn.Linear(hidden_size // 4, label_output_size)
        self.correctness_head = nn.Linear(hidden_size // 4, correctness_output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.shared_fc1(self.dropout(lstm_out[:, -1, :])))
        x = self.relu(self.shared_fc2(self.dropout(x)))
        label_output = self.label_head(x)
        correctness_output = self.correctness_head(x)
        return label_output, correctness_output