import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim = 390, layer_dim_1 = 64, layer_dim_2 = 32, output_dim = 1, p = 0.1):
        super(Model, self).__init__()
        self.layer_dim_1 = layer_dim_1
        self.input_dim = input_dim
        self.layer_dim_2 = layer_dim_2
        self.output_dim = output_dim
        
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(input_dim, layer_dim_1),
            nn.ReLU(),
        )

        self.hidden_layer_2 = nn.Sequential(
            nn.Linear(layer_dim_1, layer_dim_2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=p)
        self.output = nn.Linear(layer_dim_2 , output_dim)

    def forward(self, input):
        # input : batch_size, input_dim
        hidden_1 = self.hidden_layer_1(input)
        hidden_1 = self.dropout(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_1)
        hidden_2 = self.dropout(hidden_2)

        output = self.output(hidden_2)
        return output