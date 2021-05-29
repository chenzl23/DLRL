import torch
from data_preprocessor import *
import os
from DLRL import DLRL
import numpy as np

# Parameter settings
seed = 2020
dataset = "ml_100k"  
verbose = 10
lamda = 1e2
init_eta = 0.1
maxiter = 300
pi = torch.FloatTensor([1,1,1,1,1]) 
summ = 0.0
for p in pi:
    summ += (1 / p)
print(1 / summ)
d = 20  
lr = 0.01  

if __name__ == "__main__":
    print(dataset, lr)
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    
    # load file
    R, mask, train_mask, test_mask = read_rating(os.path.join("./data/",dataset), 0.8)


    m, n = R.size()
    model = DLRL(R, train_mask, test_mask, d, pi, maxiter, lr, init_eta, scale=1,patience=10,decay=0.1,verbose = verbose) 
    model.train()
    x_prob = model.predict()

