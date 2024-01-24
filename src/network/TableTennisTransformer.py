import torch
import yaml

class LearnedPositionalEmbeddings(torch.nn.Module):  # from ViT
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.positional_encodings = torch.nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe


class TableTennisTransformer(torch.nn.Module):
    def __init__(self, model_config: str):
        with open(model_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.seq_len = config['seq_len']
            self.hidden_dim = config['hidden_dim']
            self.num_heads = config['num_heads']
            self.num_layers = config['num_layers']
            self.dropout = config['dropout']
            self.input_dim = config['input_dim']

        super().__init__()
        self.pos_encoder = LearnedPositionalEmbeddings(self.hidden_dim, self.seq_len)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim, self.num_heads, self.hidden_dim, self.dropout)  # single encoder block
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.encoder_block = torch.nn.TransformerEncoder(encoder_layer, self.num_layers, norm=self.norm)
        self.embedder = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear = torch.nn.Linear(self.hidden_dim, 1)  # same head as in hiera, but for binary classification

        print("Num trainable parameters: " + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))


    def forward(self, x):
        # input shape (batch_size, 1, 16, 35, 2)
        x = x.squeeze()  # (batch_size, 16, 35, 2)
        x = x.transpose(2, 1)  # (batch_size, 35, 16, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, 35, 32)
        shape = x.shape
        x = self.embedder(x.view(-1, x.size(2)))  # (batch_size * 35, hidden_dim)
        x = x.reshape(shape[0], shape[1], -1).transpose(0, 1)  # (35, batch_size, hidden_dim)
        x = self.pos_encoder(x)
        x = self.encoder_block(x)  # (35, batch_size, hidden_dim)
        x = x.transpose(0, 1)  # (batch_size, 35, hidden_dim)
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        x = self.norm(x)  # (batch_size, hidden_dim)
        x = self.linear(x)  # (batch_size, 1)
        return x
