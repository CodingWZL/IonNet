import torch
import torch.nn as nn
import torch.optim as optim

class MultiLayerAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(MultiLayerAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu = nn.ReLU()
        self.query = nn.Linear(hidden_dim3, hidden_dim3)
        self.key = nn.Linear(hidden_dim3, hidden_dim3)
        self.value = nn.Linear(hidden_dim3, hidden_dim3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim3, num_heads=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        return attn_output.squeeze(0)

class MultiAttentionNetwork(nn.Module):
    def __init__(self):
        super(MultiAttentionNetwork, self).__init__()
        self.attn1 = MultiLayerAttention(120, 48, 16, 16)  
        self.attn2 = MultiLayerAttention(132, 48, 16, 16)  
        self.attn3 = MultiLayerAttention(80, 48, 16, 16)  
        self.fc = nn.Linear(16 + 16 + 16, 1)  
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(0.3))


    def forward(self, x1, x2, x3):
        out1 = self.attn1(x1)
        out2 = self.attn2(x2)
        out3 = self.attn3(x3)
        # out3 = self.attn3(x3)
        weighted_out1 = self.weight1 * out1
        weighted_out2 = self.weight2 * out2
        weighted_out3 = self.weight3 * out3
        combined = torch.cat((weighted_out1, weighted_out2, weighted_out3), dim=1) 
        output = self.fc(combined)
        return output



# Define loss and optimizer
def create_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.002)

def create_criterion():
    return nn.MSELoss()

def create_model():
    model = MultiAttentionNetwork()
    # model.apply(init_weights)
    return model



# model = MultiAttentionNetwork()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform(m.weight)
#         nn.init.zeros_(m.bias)
#
# model.apply(init_weights)




