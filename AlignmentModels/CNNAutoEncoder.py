from .AModels import AlignmentModel
import torch
from torch import nn

class Conv1dAutoencoder(AlignmentModel):
    def __init__(self, input_shape = (100,1536), output_shape = (100,1536)):
        self.input_shape = input_shape 
        self.output_shape = output_shape 

        self.input_channels = input_shape[1]
        self.output_channels = output_shape[1]

        self.input_length = input_shape[0]
        self.output_length = output_shape[0]

        super().__init__(input_shape, output_shape)
        
        # Encoder
        self.encoder = nn.Sequential(
            # First layer to reduce to (100, 800)
            nn.Conv1d(in_channels=self.input_channels, out_channels=800, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to reduce to (100, 100)
            nn.Conv1d(in_channels=400, out_channels=200, kernel_size=1),
            nn.LeakyReLU(0.1),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            
            # First layer to expand back to (100, 800)
            nn.ConvTranspose1d(in_channels=200, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose1d(in_channels=400, out_channels=800, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to expand back to (100, 1536)
            nn.ConvTranspose1d(in_channels=800, out_channels=self.output_channels, kernel_size=1),
            nn.Tanh()  # Use Tanh to output values in the range [-1, 1]
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_length*200, 2048),
            nn.Tanh()
            )
        self.fc2 =nn.Sequential(
            nn.Linear(2048, self.output_length*200),
            nn.Tanh()
            )
    def forward(self, x):
        # Transpose to (batch_size, in_channels, sequence_length)
        x = x.transpose(1, 2)  # Shape becomes (batch_size, 1536, 100)
        
        # Encode
        encoded = self.encoder(x)  # Shape becomes (batch_size, 100, 100)

        encoded_flatten = self.flatten(encoded)

        bottle_neck = self.fc1(encoded_flatten)

        decoded_fc = self.fc2(bottle_neck)

        encoded = decoded_fc.view(-1, 200, self.output_length)
        
        # Decode
        decoded = self.decoder(encoded)  # Shape becomes (batch_size, 1536, 100)
        
        # Transpose back to (batch_size, sequence_length, feature_dimension)
        decoded = decoded.transpose(1, 2)  # Final shape (batch_size, 100, 1536)
        
        return decoded

class CNN1DRBencoder(Conv1dAutoencoder):
    def __init__(self, input_shape = (100,1536), output_shape = (100,1536)):
        super().__init__(input_shape, output_shape)
        
        # Encoder
        self.encoder = nn.Sequential(
            
            nn.Conv1d(in_channels=self.input_channels, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to reduce to (100, 100)
            nn.Conv1d(in_channels=400, out_channels=200, kernel_size=1),
            nn.LeakyReLU(0.1),
        )
class CNN1DRBdecoder(AlignmentModel):
    def __init__(self, input_shape = (100,1536), output_shape = (100,1536)):
        super().__init__(input_shape, output_shape)

        self.input_channels = input_shape[1]
        self.output_channels = output_shape[1]
        self.output_length = output_shape[0]
        self.inpyt_length = input_shape[0]
        
        self.decoder = nn.Sequential(
            
            # First layer to expand back to (100, 800)
            nn.ConvTranspose1d(in_channels=200, out_channels=400, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose1d(in_channels=400, out_channels=800, kernel_size=1),
            nn.LeakyReLU(0.1),
            
            # Second layer to expand back to (100, 1536)
            nn.ConvTranspose1d(in_channels=800, out_channels=self.output_channels, kernel_size=1),
            nn.Tanh()  # Use Tanh to output values in the range [-1, 1]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_channels, self.output_length*200),
            nn.LeakyReLU(0.1)
            )
    
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 200, self.output_length)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        return x

class LSTMDecoder(AlignmentModel):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=1536, num_layers=2, seq_len=100):
        super().__init__((seq_len, input_dim), (seq_len, output_dim))
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Define an initial linear layer to expand the input to the hidden dimension
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        
        # Define the LSTM
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Define the output projection layer to map LSTM hidden states to the output dimension
        self.output_fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Expand the input vector to the LSTM hidden size
        x = self.initial_fc(x)  # Shape: [1, hidden_dim]
        
        # Repeat input along the sequence dimension for initial input to LSTM
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Shape: [1, seq_len, hidden_dim]
        
        # Pass through the LSTM
        lstm_out, _ = self.lstm(x)  # Shape: [1, seq_len, hidden_dim]
        
        # Project each LSTM output to the desired output dimension
        output = self.output_fc(lstm_out)  # Shape: [1, seq_len, output_dim]
        
        return output

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim=768, d_model=512, output_dim=1536, num_layers=2, num_heads=8, seq_len=100, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Linear layer to map the input vector to the transformer dimension
        self.input_fc = nn.Linear(input_dim, d_model)
        
        # Positional encoding to help the transformer with sequence information
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection layer to the target output dimension
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Map input to d_model dimension
        x = self.input_fc(x)  # Shape: [1, d_model]
        
        # Repeat the input vector as initial query for the transformer, adding positional encoding
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) + self.positional_encoding  # Shape: [1, seq_len, d_model]
        
        # Prepare the target sequence (the queries for the Transformer Decoder)
        tgt = torch.zeros(self.seq_len, x.size(0), self.d_model, device=x.device)  # Shape: [seq_len, batch_size, d_model]

        # Transformer expects [target_seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # Shape: [seq_len, batch_size, d_model]
        
        # Pass through the Transformer decoder
        transformer_out = self.transformer_decoder(tgt=tgt, memory=x)  # Shape: [seq_len, batch_size, d_model]

        # Project each output to the desired output dimension
        output = self.output_fc(transformer_out)  # Shape: [seq_len, batch_size, output_dim]
        
        # Transpose to match desired output shape: [batch_size, seq_len, output_dim]
        output = output.permute(1, 0, 2)  # Shape: [1, seq_len, output_dim]
        
        return output