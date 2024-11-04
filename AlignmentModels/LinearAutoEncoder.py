# We define a simple linear autoencoder model for alignment
from .AModels import AlignmentModel
import torch.nn as nn

class LinearAlignementModel(AlignmentModel):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        input_dim = input_shape[0] * input_shape[1]
        output_dim = output_shape[0] * output_shape[1]
        self.flatten = nn.Flatten()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024*2),  # Input size: 200*1536, Output size: 1024
            nn.LeakyReLU(0.1),
            nn.Linear(1024*2,1024),
            # nn.LeakyReLU(0.001),
            # nn.Linear(1024*2, 1024)             # Bottleneck size: 64
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(1024, 1024*2),             # Input size: 64, Output size: 256
            # nn.LeakyReLU(0.001),
            nn.Linear(1024,1024*2),
            nn.LeakyReLU(0.1),
            nn.Linear(1024*2, output_dim),    # Output size: 200*1536
            nn.Tanh()                  #
        )
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1,self.output_shape[0],self.output_shape[1])   # Reshape to original image size