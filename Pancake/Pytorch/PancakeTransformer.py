import torch
import torch.nn as nn
import torch.nn.functional as F

class PancakeTransformer(nn.Module):
    def __init__(self, game, args):
        # game params
        self.input_size = 2
        self.action_size = game.getActionSize()
        self.args = args

        super(PancakeTransformer, self).__init__()
        # Define the transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=2, nhead=2, dropout=0.1)

# Stack two layers on top of each other
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)        
        #self.transformer = nn.Transformer(d_model=2, nhead=8, num_layers=2, dropout=0.1)
        self.fc1 = nn.Linear(game.n*2,self.action_size)
        self.fc2 = nn.Linear(512, self.action_size)
        self.fc3 = nn.Linear(game.n*2, 1)
        self.reward = nn.ELU()

    def forward(self, s):
        #s = s.view(-1, self.input_size, 1)
        s = self.transformer(s)
        s = s.reshape(s.shape[0], -1)
        pi = F.relu(self.fc1(s))
        #pi = self.fc2(s)
        v = self.reward(self.fc3(s))
        return F.log_softmax(pi, dim=1), v
