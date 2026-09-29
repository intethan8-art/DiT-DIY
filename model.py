import torch
import torch.nn as nn
import torch.nn.functional as F

# naive using MLP

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, t):
        inputs = torch.cat((x, t.reshape(-1, 1)), 1)
        x1 = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out
        
        