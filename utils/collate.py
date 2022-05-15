# this is for padding with maximum data in a batch
import numpy as np
import torch
from .dataset import MAX_IMAGE_SIZE


def collate_fn(batch):
    # zero pad to max size, so make equal size in the batch
    
    '''
    # this is for 1d case, just example
    longest = max([len(x) for x in batch])
    s = np.stack([np.pad(x, (0, longest - len(x))) for x in batch])
    return torch.from_numpy(s)
    '''
    labels = [ b[1] for b in batch]
    labels = torch.tensor(labels, dtype=torch.uint8)
    
    # for testing area based batch
    # areas = [ b[2] for b in batch]
    # areas = torch.tensor(areas, dtype=torch.int)
    
    max_x = max([ b[0].shape[0] for b in batch])
    max_y = max([ b[0].shape[1] for b in batch])
    
    # zero pad to right, bottom side
    s_batch = np.stack([np.pad(b[0], ((0, max_x - b[0].shape[0]), (0, max_y - b[0].shape[1]), (0,0))) for b in batch])    
    
    '''
    # if variation is big, losts of padding is added to small size image, then maybe need to crop also
    min_x = min([ b.shape[0] for b in batch])
    min_y = min([ b.shape[1] for b in batch])
    
    # select approriate size
    # a = int(max_x - min_x) * 0.8
    # b = int(max_y - min_y) * 0.8
    # s_batch = s_batch[:,:a,:b,:]
    '''
    
    # crop if image is larger than MAX_IMAGE_SIZE at once
    s_batch = s_batch[:,:MAX_IMAGE_SIZE,:MAX_IMAGE_SIZE,:]
    
    # transpose to pytorch format
    s_batch = np.transpose(s_batch, (0,3,1,2))
    
    # change dtype to torch.float
    s_batch = torch.tensor(s_batch, dtype=torch.float32)
    
    return s_batch, labels
    # return s_batch, labels, areas
    
    
    
    