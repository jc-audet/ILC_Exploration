import torch
from torch import nn as nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
        
def get_mnist_model(args, device):
    if args.n_hidden_layers >= 1:
        hidden_layers = [torch.nn.Linear(args.n_dims, args.n_hidden_units)]
        for _ in range(args.n_hidden_layers - 1):
            hidden_layers.append(torch.nn.Linear(args.n_hidden_units, args.n_hidden_units))
        classification_layer = torch.nn.Linear(args.n_hidden_units, args.n_outputs)
        layers = []
        for linear_layer in hidden_layers:
            layers.append(linear_layer)
            if args.batch_norm:
                layers.append(nn.BatchNorm1d(args.n_hidden_units))
            layers.append(nn.LeakyReLU())
            if args.dropout_p > 0.0:
                layers.append(nn.Dropout(p=args.dropout_p))
        layers.append(classification_layer)
        model = torch.nn.Sequential(*layers).to(device)
    else:
        model = LogisticRegression(args.n_dims, args.n_outputs)
        model.to(device)
    return model