import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, 
                 embedding_layer: nn.Embedding, 
                 hidden_dim: int, 
                 n_layers: int, 
                 bidirectional: bool, 
                 dropout: float, 
                 output_dim: int = 6, 
                 rnn_type: str = 'lstm'):
        """
        The main Emotion Classification Model.
        
        Args:
            embedding_layer: Pre-trained or trainable embedding layer.
            hidden_dim: Size of the hidden state.
            n_layers: Number of RNN layers.
            bidirectional: If True, uses Bi-LSTM/Bi-GRU.
            dropout: Dropout probability.
            output_dim: Number of classes (emotions).
            rnn_type: 'lstm' or 'gru'.
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # 1. Embedding Layer
        self.embedding = embedding_layer
        embedding_dim = embedding_layer.embedding_dim
        
        # 2. RNN Layer (LSTM or GRU)
        rnn_args = {
            'input_size': embedding_dim,
            'hidden_size': hidden_dim,
            'num_layers': n_layers,
            'bidirectional': bidirectional,
            'dropout': dropout if n_layers > 1 else 0,
            'batch_first': True
        }
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_args)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_args)							   
        else:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Supported: 'lstm', 'gru'")
        
        # 3. Fully Connected Layer
        # Calculate input size for linear layer based on directionality
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # 4. Dropout (Applied to embedding output and rnn output)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths=None):
        """
        Forward pass.
        Args:
            text: [batch_size, seq_len]
            text_lengths: [batch_size] (Actual lengths of sequences, optional usage)
        """
        # Apply dropout to embeddings
        embedded = self.dropout(self.embedding(text)) # [batch_size, sent_len, emb_dim]
        # RNN pass
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)
            
        # Extract the final hidden state
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            # hidden shape: [num_layers * num_directions, batch, hidden_size]
            hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Take the last hidden state
            hidden_final = hidden[-1,:,:]
            
        # Apply dropout to the hidden state
        hidden_final = self.dropout(hidden_final)
            
        # Classification
        prediction = self.fc(hidden_final) # [batch_size, output_dim]
        
        return prediction