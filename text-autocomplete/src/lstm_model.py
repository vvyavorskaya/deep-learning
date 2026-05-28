import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

    
    def predict_next_tokens(self, input_text, vocab, num_tokens=10):
        self.eval()
        tokens = input_text.split()
        input_ids = [vocab['word_to_idx'].get(token, 1) for token in tokens]
        
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for _ in range(num_tokens):
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                output = self.forward(input_tensor)
                next_token_id = torch.argmax(output[0, -1, :]).item()
                
                if next_token_id == 0 or len(tokens) > 20:
                    break
                    
                input_ids.append(next_token_id)
                next_token = vocab['idx_to_word'].get(next_token_id, '<UNK>')
                tokens.append(next_token)
        
        return ' '.join(tokens)