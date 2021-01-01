import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.l1 = nn.Linear(9, 20)
        self.l2 = nn.Linear(20, 9)
        self.l3 = nn.Linear(20,1)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x1 = self.l2(x) # Logit value of policy
        x2 = torch.tanh(self.l3(x)) # Value head.
        return x1, x2

    def eval(self,x):
        with torch.no_grad():
            p, v = self.forward(x)

        p = self.sm(p.view(-1,9))
        if p.size(0)==1:
            return p[0,:], v[0]


        return p, v

    def save_checkpoint(self, name='checkpoint.pth'):
        torch.save(self.state_dict(), name)
