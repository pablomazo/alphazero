import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        # Net structure from https://codebox.net/pages/connect4
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, (4,4)) # Padding to keep same input size.
        self.l1 = nn.Linear(1536, 164)
        self.l2 = nn.Linear(164, 64)
        self.l3 = nn.Linear(64, 7)
        self.l4 = nn.Linear(64, 1)
        self.flatten = nn.Flatten()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 6, 7)))
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x1 = self.l3(x) # Logit value of policy
        x2 = torch.tanh(self.l4(x)) # Value head.

        return x1, x2

    def eval(self,x):
        with torch.no_grad():
            p, v = self.forward(x)

        p = self.sm(p)
        if p.size(0)==1:
            return p[0,:], v[0,:]

        return p, v

    def save_checkpoint(self, name='checkpoint.pth'):
        torch.save(self.state_dict(), name)
