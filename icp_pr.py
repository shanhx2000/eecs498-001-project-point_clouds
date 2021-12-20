#!/usr/bin/env python
from os import error
from numpy.core.defchararray import decode
from numpy.lib.twodim_base import eye
import hw4.utils as utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
from numpy.linalg import svd, norm
from PointCloudFeature import PointCloudFeature as PCF
import gc
import time
from copy import copy
import random

###YOUR IMPORTS HERE###
def GetTranform( Cp, Cq ):
    p = np.mean(Cp, axis=1).reshape( (3,1) )
    q = np.mean(Cq, axis=1).reshape( (3,1) )
    X = Cp - p
    Y = Cq - q
    S = X @ Y.T
    u,sig,v = np.linalg.svd(S)
    v = v.T
    tmp = np.eye(3)
    tmp[2,2] = np.linalg.det(v@u.T)
    R = v@tmp@u.T
    t = q - R@p
    return R,t

def Join_Feature_Set(source, target):
    source_list = source.p_f_list
    target_list = target.p_f_list
    # print("source_list", np.array(source.p_f_list).shape)
    # print("target_list", target_list.shape)
    Cp = []
    Cq = []
    start = time.time()
    target_list_np = np.array(target_list)
    for s in source_list:
        vmin = 1e5
        tmp_p = None
        
        for t in target_list:
            if norm(t.feature-s.feature) < vmin:
                vmin = norm(t.feature-s.feature)
                tmp_p = t.point
        assert tmp_p is not None
        Cp.append(s.point.reshape( (3,1) ))
        Cq.append(tmp_p.reshape( (3,1) ))
    # print("JOIN_FEATURE_SET =", time.time() - start)
    return Cp, Cq

def get_boundary(PC):
    """
    Return the upper and lower bound of each dimension.
    """
    return np.max(PC, axis=0), np.min(PC, axis=0)

def get_sparse_PC(PC, target_N):
    """
    Filter out concentrated data; Guarantee that the generated PC has the size of `target_N`.
    """
    ratio = 20
    upper_bound, lower_bound = get_boundary(PC)
    # Generate max, min, and diff for each dimension
    max_x, max_y, max_z = upper_bound[0], upper_bound[1], upper_bound[2]
    min_x, min_y, min_z = lower_bound[0], lower_bound[1], lower_bound[2]
    # diff is the side length of each small cube
    diff_x, diff_y, diff_z = (max_x-min_x) / ratio, (max_y-min_y) / ratio, (max_z-min_z) / ratio
    count = np.zeros((ratio, ratio, ratio))
    # point_dict records which cube one point belongs to
    point_dict = {}
    for i in range(PC.shape[0]):
        # Count the number of points in each block
        x = (int)((PC[i][0]-min_x) / diff_x)
        y = (int)((PC[i][1]-min_y) / diff_y)
        z = (int)((PC[i][2]-min_z) / diff_z)
        # print(x,y,z)
        x = min(ratio-1, x)
        y = min(ratio-1, y)
        z = min(ratio-1, z)
        count[x][y][z] += 1
        if str(x)+','+str(y)+','+str(z) in point_dict:
            point_dict[str(x)+','+str(y)+','+str(z)].append(i)
        else:
            point_dict[str(x)+','+str(y)+','+str(z)] = [i]
    count = np.ceil(count / PC.shape[0] * target_N)
    while (np.sum(count) != target_N):
        if np.sum(count) > target_N:
            # Pick one non-zero entry to reduce 1 point
            while True:
                rand_index = np.random.rand(3,1) * ratio
                if count[(int)(rand_index[0])][(int)(rand_index[1])][(int)(rand_index[2])] > 0:
                    count[(int)(rand_index[0])][(int)(rand_index[1])][(int)(rand_index[2])] -= 1
                    break
        else:
            # Pick one still available entry to add 1 point
            while True:
                rand_index = np.random.rand(3,1) * ratio
                x = (int)(rand_index[0])
                y = (int)(rand_index[1])
                z = (int)(rand_index[2])
                if str(x)+','+str(y)+','+str(z) in point_dict and count[x][y][z] < len(point_dict[str(x)+','+str(y)+','+str(z)]):
                    count[x][y][z] += 1
                    break
    target_PC = np.zeros((target_N, 3))
    index = 0
    for i in range(ratio):
        for j in range(ratio):
            for k in range(ratio):
                if count[i][j][k] == 0:
                    continue
                tmp_point_index_list = random.sample(point_dict[str(i)+','+str(j)+','+str(k)], int(count[i][j][k]))
                for point_index in tmp_point_index_list:
                    target_PC[index] = PC[point_index]
                    index += 1
    return target_PC


def main():
    #Import the cloud
    start = time.time()
    N = 200000
    PC1 = np.loadtxt('data_pcd/capture0003.txt')[:N]
    PC2 = np.loadtxt('data_pcd/capture0001.txt')[:N]


    tmp_PC1 = get_sparse_PC(PC1, int(N / 10000))
    print(PC1.shape, tmp_PC1.shape)
    
    tmp_PC2 = get_sparse_PC(PC2, int(N / 10000))
    print(tmp_PC1.shape)
    print("After get sparse PC:", time.time() - start)

    # tmp_PC2 = tmp_PC2 + 1
    P1 = np.expand_dims(tmp_PC1, axis = 2)
    P2 = np.expand_dims(tmp_PC2, axis = 2)
    
    P = utils.convert_pc_to_matrix(P1)
    Q = utils.convert_pc_to_matrix(P2)
    print ( P.shape )
    doneFlag = False
    bestCost = 99999999
    eps = 1e-3 # 1e-2
    itr = 0
    error_list = []
    while ( not doneFlag and itr < 500 ):
        Cp = []
        Cq = []
        feature_p = PCF(P, verbose=True)
        feature_q = PCF(Q)
        feature_p.build_features()
        feature_q.build_features()
        Cp, Cq = Join_Feature_Set(feature_p, feature_q)

        Cp = np.squeeze(np.array(Cp),axis=2).T
        Cq = np.squeeze(np.array(Cq),axis=2).T
        R,t = GetTranform( Cp, Cq )

        del Cp, Cq, feature_p, feature_q
        gc.collect()

        # P = R@Cp+t
        P = R@P+t # Newly, Should be this one
        newCost = np.sum( np.linalg.norm(R@P+t-Q , axis=0)**2  )
        error_list.append(newCost)

        if ( newCost < bestCost ):
            bestCost = newCost
            print ( itr , " : ", bestCost )
        if ( abs(newCost)  < eps):
            doneFlag = True

        if ( itr > 10 and abs(error_list[-5] - newCost) < eps):
            print ( error_list[-5] - newCost )
            doneFlag = True

        itr = itr + 1
        if ( itr % 20 == 0 ):
            print ( "Iteration" , itr )

    print ( P.shape )
    print("Total Time:", time.time() - start)
    pc_fit = utils.convert_matrix_to_pc( np.expand_dims(P,axis=2) )
    # print(pc_fit.shape, tmp_PC1.shape, tmp_PC2.shape)
    utils.view_pc([pc_fit, P1, P2], None, ['b', 'r', 'g'], ['o', '^', 'o'])

    plt.axis([-2, 2, -2, 2])

    fig1 = plt.figure()
    plt.plot( range(len(error_list)) , error_list )
    plt.show()


if __name__ == '__main__':
    main()
