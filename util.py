import torch

def Spectral_norm(x):
    m, n = x.size()
    if 2*m < n:
        try:
            s = torch.sqrt(torch.svd(x.matmul(x.t()))[1][0])
        except RuntimeError:
            print(x)
        return s
    if m > 2*n:
        s = Spectral_norm(x.t())
        return s
    return torch.svd(x)[1][0]

def schatten_p_norm(x, p):
    u, s, v = torch.svd(x)
    sp = torch.sum(s ** p)
    return sp
