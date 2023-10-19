#! /usr/bin/env python3
#
# Drop this file in the ~/SuperGluePretrainedNetwork/models/weights/ of https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth 
# and ~/SuperGluePretrainedNetwork/models/ of https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py
# To use with rtabmap:
#   cp rtabmap_lightglue.py ~/SuperGluePretrainedNetwork/
#
#                 

import random
import numpy as np
import torch

#import sys
#import os
#print(os.sys.path)
#print(sys.version)

from models.lightglue import LightGlue

torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

matcher = LightGlue(features='superpoint').eval().to(device)

def init(descriptorDim, matchThreshold, iterations, cuda, model):
    print("LightGlue python init()")
    # Load the SuperGlue model.
    global device
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'

    global matcher
    matcher = LightGlue(features=None, weights = "superpoint_lightglue").eval().to(device)


def match(kptsFrom, kptsTo, scoresFrom, scoresTo, descriptorsFrom, descriptorsTo, imageWidth, imageHeight):
    #print("SuperGlue python match()")
    global device
    kptsFrom = np.asarray(kptsFrom)
    kptsFrom = kptsFrom[None, :, :]
    kptsTo = np.asarray(kptsTo)
    kptsTo = kptsTo[None, :, :]
    scoresFrom = np.asarray(scoresFrom)
    scoresFrom = scoresFrom[None, :]
    scoresTo = np.asarray(scoresTo)
    scoresTo = scoresTo[None, :]
    descriptorsFrom = np.transpose(np.asarray(descriptorsFrom))
    descriptorsFrom = descriptorsFrom[None, :, :]
    descriptorsTo = np.transpose(np.asarray(descriptorsTo))
    descriptorsTo = descriptorsTo[None, :, :]
      
    data = {
        'image0':{
            'keypoints': torch.from_numpy(kptsFrom).permute(0, 2, 1).to(device),
            'keypoints_scores': torch.from_numpy(scoresFrom).to(device),\
            'descriptors': torch.from_numpy(descriptorsFrom).permute(0, 2, 1).to(device),
            'image_size': torch.tensor([[imageHeight, imageWidth]]).to(device)

        },

        'image1':{
            'keypoints': torch.from_numpy(kptsTo).permute(0, 2, 1).to(device),
            'keypoints_scores': torch.from_numpy(scoresTo).to(device),
            'descriptors': torch.from_numpy(descriptorsTo).permute(0, 2, 1).to(device),
            'image_size': torch.tensor([[imageHeight, imageWidth]]).to(device)
        },
    }
    
# TODO: fix bug while lightglue mathing with less superpoints kpts 

    global matcher
    results = matcher(data)

    matches0 = results['matches'].to('cpu').numpy()
  
    matchesFrom = np.nonzero(matches0!=-1)[1]
    matchesTo = matches0[np.nonzero(matches0!=-1)]
       
    matchesArray = np.stack((matchesFrom, matchesTo), axis=1)
    
    return matchesArray


if __name__ == '__main__':
    #test
    init(256, 0.2, 20, True, 'indoor')
    match([[1, 2], [1,3]], [[1, 3], [1,2]], [1, 3], [1,3], np.full((2, 256), 1),np.full((2, 256), 1), 640, 480)