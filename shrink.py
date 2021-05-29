import torch
from torch import nn
import torch.nn.functional as F

class SHRINK(nn.Module):
    def __init__(self, block_num = 1):
        super(SHRINK, self).__init__()
        self.block_num = block_num
        
    def forward(self, x, W, p):
        m,n = x.size()
        if m < 2*n: 
            x = x.t()
            W = W.t()
        new_x = []
        # avoid zeros
        st = 0.002*max(abs(torch.diag(x.t().matmul(x))))
        x = F.relu(x)
        D = (1/2) * torch.pow((x.t().matmul(x)) + st, (p-2)/2) 
        for i in range(self.block_num):
            temp_x = 0.5 * W.matmul(torch.pinverse(D))
            temp_x = F.relu(temp_x)
            D = (1/2) * torch.pow((temp_x.t().matmul(temp_x)) + st, (p-2)/2)
            new_x.append(temp_x)

        if m < 2*n:
            return_x = torch.FloatTensor(new_x[-1]).t()
        else:
            return_x = torch.FloatTensor(new_x[-1])
        return return_x
