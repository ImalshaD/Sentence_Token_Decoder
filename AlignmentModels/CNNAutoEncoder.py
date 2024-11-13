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

