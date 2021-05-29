import torch
import random
from tqdm import tqdm
import os

def read_rating(path, train_ratio):
    print(path)
    fp = open(os.path.join(path, "ratings.txt"))
    random.seed(2019)
    lines = fp.readlines()

    num_users = 0
    num_items = 0
    num_total_ratings = len(lines)
    for line in lines:
        user,item,rating,_ = line.split("::")
        user_idx = int(user)
        item_idx = int(item)
        num_users = max(num_users, user_idx)
        num_items = max(num_items, item_idx)
    print("number of users:", num_users, "number of items:", num_items, "number of ratings:", num_total_ratings)


    R = torch.zeros((num_users,num_items))
    mask_R = torch.zeros((num_users, num_items))


    train_mask_R = torch.zeros((num_users, num_items))
    test_mask_R = torch.zeros((num_users, num_items))

    random_perm_idx = torch.randperm(num_total_ratings)

    train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]


    for line in lines:
        user,item,rating,_ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx,item_idx] = float(rating)
        mask_R[user_idx,item_idx] = 1

    ''' Train '''
    for itr in tqdm(train_idx,desc="train data processing"):
        line = lines[itr]
        user,item,rating,_ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_mask_R[user_idx,item_idx] = 1

    ''' Test '''
    for itr in tqdm(test_idx,desc="test data processing"):
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_mask_R[user_idx, item_idx] = 1

    return R, mask_R, train_mask_R, test_mask_R
