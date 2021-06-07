import torch
from torch import nn
from util import Spectral_norm, schatten_p_norm
from shrink import SHRINK
import time

class DLRL(nn.Module):
    def __init__(self, M, I, I_test, d, p, maxiter, lr, eta, scale,patience, decay, verbose):
        super(DLRL, self).__init__()
        self.M = M
        self.I = I
        self.I_test = I_test
        self.maxiter = maxiter
        self.lr = lr
        self.p = p
        self.d = d
        self.patience = patience
        self.decay = decay
        self.verbose = verbose

        self.eta = nn.Parameter(torch.FloatTensor([eta]), requires_grad = True)  
        
        self.p_len = p.size(0)
        self.m, self.n = M.size()

        self.X = []
        self.X.append(scale*torch.rand(self.m, self.d))
        for i in range(1, self.p_len-1):
            self.X.append(torch.eye(self.d, self.d))
        self.X.append(scale*torch.rand(self.d, self.n))


        self.W = nn.ParameterList()
        self.W.append(nn.Parameter(torch.rand(self.m, self.d), requires_grad=True)) 
        for i in range(1, self.p_len-1):
            self.W.append(nn.Parameter(torch.eye(self.d, self.d), requires_grad=True))  
        self.W.append(nn.Parameter(torch.rand(self.d, self.n), requires_grad=True))

        self.prox = SHRINK()  


    def forward(self, u):
        x_list = u
        return_x = []
        eta = min(max(self.eta, 0), 1)
        for i in range(self.p_len): 
            ind = i % self.p_len
            x = x_list[ind]
            if ind == 0:
                x_left = torch.eye(self.m, self.m)
                x_right = torch.eye(self.d, self.d)
                for j in range(1, self.p_len):
                    x_right = x_right.matmul(x_list[j]) 
                lipz_temp = max(Spectral_norm(x_right), torch.FloatTensor([1e-4]))
                lipz = eta * (lipz_temp ** 2)
            elif ind == (self.p_len-1):
                x_left = torch.eye(self.m, self.m)
                x_right = torch.eye(self.n, self.n)
                for j in range(0, self.p_len - 1):
                    x_left = x_left.matmul(x_list[j]) 
                lipz_temp = max(Spectral_norm(x_left), torch.FloatTensor([1e-4]))
                lipz = eta * (lipz_temp ** 2)
            else:
                x_left = torch.eye(self.m, self.m)
                x_right = torch.eye(self.d, self.d)
                for j in range(0, ind):
                    x_left = x_left.matmul(x_list[j]) 
                for j in range(ind + 1, self.p_len):
                    x_right = x_right.matmul(x_list[j])
                lipz_templ = max(Spectral_norm(x_left), torch.FloatTensor([1e-4]))
                lipz_tempr = max(Spectral_norm(x_right), torch.FloatTensor([1e-4]))
                lipz = eta * (lipz_templ ** 2) * (lipz_tempr ** 2)

            x_prob = x_left.matmul(x).matmul(x_right)
            WXM = self.I.mul(self.M - x_prob)
            delta_f = x_left.t().matmul(WXM).matmul(x_right.t()) / lipz
            temp_x_ = x - delta_f
            temp_x = self.prox(temp_x_, self.W[ind], self.p[ind])  
            return_x.append(temp_x)
        return return_x

    def my_loss(self, M, X):
        x_prob = torch.eye(self.m, self.m)
        for i in range(self.p_len):
            x_prob = x_prob.matmul(X[i])
        return 0.5 * (torch.norm((M - x_prob).mul(self.I))**2)
    

    def compute_sp_norm(self, x_list):
        sp_sum = 0.0
        for ind in range(len(x_list)):
            sp = schatten_p_norm(x_list[ind], self.p[ind]) / self.p[ind]
            sp_sum += sp
        return sp_sum.cpu().detach().numpy()

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,betas=(0.90, 0.92),weight_decay=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = self.decay, patience = self.patience, verbose = True, min_lr=1e-4 * self.lr) # learning rate decay

        start = time.perf_counter()
        for epoch in range(self.maxiter):
            x_list = self(self.X)
            loss = self.my_loss(self.M, x_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.cpu().detach().numpy()
            scheduler.step(loss)
            if (optimizer.param_groups[0]['lr'] < 1e-3 * self.lr):
                print("early stopped")
                break
            if epoch % self.verbose == 0 and epoch != 0:
                print("epoch " + str(epoch))
                print("loss: {: .4f}".format(train_loss))
                elapsed = (time.perf_counter() - start)
                print("Time used:",elapsed)
                print("--------------------------------------------------------")

    def predict(self):
        X = self(self.X)
        x_prob_all = torch.eye(self.m, self.m)
        for i in range(self.p_len):
            x_prob_all = x_prob_all.matmul(X[i])
        x_prob = x_prob_all.mul(self.I_test)
        x = self.M.mul(self.I_test)

        rmse = torch.sum((x - x_prob) ** 2) / torch.sum(self.I_test)

        print("Test RMSE:",rmse.cpu().detach().numpy())

        return x_prob_all

    