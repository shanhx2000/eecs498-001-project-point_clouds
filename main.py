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


cat = np.loadtxt('data_pcd/ism_train_wolf.txt')
theta = 0.3
R = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
t = np.array([[5, -10, 3]])


with open('data_pcd/ism_train_wolf_target.npy', 'wb') as f:
    np.save(f, (R.dot(cat.T).T+t))

with open('data_pcd/ism_train_wolf_source.npy', 'wb') as f:
    np.save(f, cat)



