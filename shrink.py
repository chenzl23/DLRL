import torch
from torch import nn
import torch.nn.functional as F

class SHRINK(nn.Module):
    def __init__(self):
        super(SHRINK, self).__init__()
        
    def forward(self, Y, W, p):
        m,n = Y.size()
        if m < 2*n: 
            Y = Y.t()
            W = W.t()

        # avoid zeros
        st = 0.002*max(abs(torch.diag(Y.t().matmul(Y))))
        G = (1/2) * torch.pow((Y.t().matmul(Y)) + st, (p-2)/2) 
        T = 0.5 * W.matmul(torch.pinverse(G))
        X = F.relu(T)

        if m < 2*n:
            return_x = torch.FloatTensor(X).t()
        else:
            return_x = torch.FloatTensor(X)
        return return_x
