import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, (5,5), padding=(2,2)) # Padding to keep same input size.
        self.l1 = nn.Linear(210, 20)
        self.l2 = nn.Linear(20, 7)
        self.l3 = nn.Linear(20, 1)
        self.flatten = nn.Flatten()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 6, 7)))
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x1 = self.l2(x) # Logit value of policy
        x1 = self.sm(x1)
        x2 = torch.tanh(self.l3(x)) # Value head.

        if x1.size(0)==1:
            return x1[0,:], x2[0,:]
        return x1, x2
