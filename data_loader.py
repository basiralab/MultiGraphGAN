import torch.utils.data as data_utils
import numpy as np
import torch
import SIMLR
import os

def get_loader(features, batch_size, num_workers=1):
    """
    Build and return a data loader.
    """
    dataset = data_utils.TensorDataset(torch.Tensor(features))
    loader = data_utils.DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle = True, #set to True in case of training and False when testing the model
                        num_workers=num_workers
                        )
    
    return loader

def learn_adj(x):
    y = []
    for t in x:
            b = t.numpy()
            y.append(b)
    
    x = np.array(y)
    batchsize = x.shape[0]
    simlr = SIMLR.SIMLR_LARGE(1, batchsize/3, 0)
    adj, _,_, _ = simlr.fit(x)
    array = adj.toarray()
    tensor = torch.Tensor(array)
    
    return tensor

def to_tensor(x):
    y = []
    for t in x:
            b = t.numpy()
            y.append(b)
    
    x = np.array(y)
    x = x[0]
    tensor = torch.Tensor(x)
    
    return tensor

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)