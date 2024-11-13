from .AModels import AlignmentModel
import torch
import torch.nn as nn

class lstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(lstmEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)
        # output shape: (batch_size, seq_len, hidden_size)
        # hidden and cell shapes: (num_layers, batch_size, hidden_size)
        return output, (hidden, cell)

class lstmDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(lstmDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, prev_state):
        # x shape: (batch_size, 1, output_size)
        # prev_state shapes: (num_layers, batch_size, hidden_size)
        output, state = self.lstm(x, prev_state)
        # output shape: (batch_size, 1, hidden_size)
        # state shapes: (num_layers, batch_size, hidden_size)
        output = self.fc(output[:, -1, :])
        # output shape: (batch_size, output_size)
        return output, state

class Seq2seqARAE(AlignmentModel):
    
    def __init__(self, input_shape = (100,1536), output_shape = (100,1536)):
        self.input_shape = input_shape 
        self.output_shape = output_shape 

        self.input_channels = input_shape[1]
        self.output_channels = output_shape[1]

        self.input_length = input_shape[0]
        self.output_length = output_shape[0]

        super().__init__(input_shape, output_shape)
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        
        # Get the cell state from the encoder
        cell = self.encoder(source)
        
        # Use the encoder's cell state to initialize the decoder
        decoder_input = torch.zeros(batch_size, 1, target.size(-1), device=source.device)
        
        outputs = []
        for t in range(target.size(1)):
            decoder_output, cell = self.decoder(decoder_input, cell)
            outputs.append(decoder_output)
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else decoder_output
        
        # Stack the outputs into a single tensor
        outputs = torch.stack(outputs, dim=1)
        # outputs shape: (batch_size, seq_len, output_size)
        
        return outputs