#!/usr/bin/env python
from os import error
from numpy.core.defchararray import decode, title
from numpy.lib.twodim_base import eye
import hw4.utils as utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
from numpy.linalg import svd, norm
from PointCloudFeature import PointCloudFeature as PCF
import gc
from pcloader import load_pcl, get_sparse_PC
import time
from PointSelector import Kmeans_select


cat = np.loadtxt('data_pcd/ism_train_cat.txt')

R = np.array([[1,0,0],[0,np.cos(0.1), -np.sin(0.1)],[0,np.sin(0.1),np.cos(0.1)]])
print(R)

t = np.array([[1, 2, -1]])

print(t)

with open('data_pcd/ism_train_cat_target.npy', 'wb') as f:
    np.save(f, (R.dot(cat.T).T+t))

with open('data_pcd/ism_train_cat_source.npy', 'wb') as f:
    np.save(f, cat)



