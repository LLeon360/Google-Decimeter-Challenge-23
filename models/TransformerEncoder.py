import torch

# time sequence encoder
class TransformerEncoder(torch.nn.module):
    def __init__(self, config):
        self.config = config
        self.embeddings = torch.nn.Linear(config.input_dim, config.d_model)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation
            ),
            num_layers=config.num_layers
        )
        self.fc = torch.nn.Linear(config.d_model, config.output_dim, activation="softmax")
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x