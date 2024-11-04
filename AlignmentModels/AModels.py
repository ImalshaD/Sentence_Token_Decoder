from abc import ABC, abstractmethod
import numpy as np

import torch.nn as nn

class AlignmentModel(ABC, nn.Module):
    input_shape : tuple
    output_shape : tuple

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, x):
        pass

    def test(self):
        sample_input  = np.zeros(self.input_shape)
        output = self.forward(sample_input)
        assert output.shape == self.output_shape