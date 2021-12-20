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
from test_PFH import PFH
from copy import copy
# from pcloader
# import pcl

###YOUR IMPORTS HERE###
def GetTranform( Cp, Cq ):
    p = np.mean(Cp, axis=1).reshape( (3,1) )
    q = np.mean(Cq, axis=1).reshape( (3,1) )
    X = Cp - p
    Y = Cq - q
    S = X @ Y.T
    u,sig,v = np.linalg.svd(S)
    v = v.T ## DEBUG
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
    # for i in range(len(source_list)):
    #     Cp.append(source_list[i].point.reshape( (3,1) ))
    #     Cq.append(target_list[i].point.reshape( (3,1) ))

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
    return Cp, Cq


def main():
    #Import the cloud
    # pc_source = utils.load_pc('hw4/cloud_icp_source.csv')
    # pc_target = utils.load_pc('hw4/cloud_icp_target0.csv') # Change this to load in a different target
    # P = utils.convert_pc_to_matrix(pc_source)
    # Q = utils.convert_pc_to_matrix(pc_target)

    # N = 20000 #200000
    # PC1 = np.loadtxt('data_pcd/capture0003.txt')[:N]
    # PC2 = np.loadtxt('data_pcd/capture0001.txt')[:N]
    # N = 500 #200000
    # PC1 = np.loadtxt('data_pcd/capture0003.txt')[:N]
    # PC2 = np.loadtxt('data_pcd/capture0001.txt')[:N]
    with open('data_pcd/ism_train_horse_source.npy','rb') as f:
        PC1 = np.load(f)
    with open('data_pcd/ism_train_horse_target.npy','rb') as f:
        PC2 = np.load(f)
    P = PC1.T
    Q = PC2.T

    print(P.shape)
    print(Q.shape)

    print("size=", P.shape)

    doneFlag = False
    bestCost = 99999999
    eps = 1e-3 # 1e-2
    itr = 0
    error_list = []

    filter_time = []
    feature_gen_time = []
    selected_points_num = []
    transform_time_total = 0
    join_time_total = 0

    st_time = time.time()

    P = np.array(P)
    Q = np.array(Q)
    all_P = copy(P)
    tmp_st = time.time()
    P, Q = Kmeans_select(P, Q, ratio=0.3)
    filter_time.append(time.time()-tmp_st)
    ori_P = copy(P)
    # P, Q = Kmeans_select(P, Q, ratio=0.001)


    print("Build Q=", Q.shape)
    tmp_st = time.time()
    feature_q = PCF(Q)
    fq = feature_q.build_features()
    feature_gen_time.append(time.time()-tmp_st)
    
    print("Start ICP!")

    while ( not doneFlag and itr < 100 ):
        Cp = []
        Cq = []
        
        tmp_st = time.time()
        # P_filterred, Q_filterred = Kmeans_select(P, Q)
        P_filterred, Q_filterred = P, Q

        selected_points_num.append(P_filterred.shape[-1])
        filter_time.append(time.time()-tmp_st)
        print("Generate ", P_filterred.shape[-1], " Points for P")

        tmp_st = time.time()
        feature_p = PCF(P_filterred, verbose=False)
        fp = feature_p.build_features()
        feature_gen_time.append(time.time()-tmp_st)
        
        assert feature_p != []

        tmp_st = time.time()
        # Cp, Cq = Join_Feature_Set(feature_p, feature_q)
        for i in range(P.shape[1]):
            pp = np.array(P[:,i]).reshape( (3,1) )
            p = fp[i,:]
            q_idx = np.argmin( np.linalg.norm(fq-p,axis=1) )
            q = np.array(Q[:,q_idx]).reshape( (3,1) )
            Cp.append( pp )
            Cq.append( q )

        join_time_total += time.time() - tmp_st
        

        Cp = np.squeeze(np.array(Cp),axis=2).T
        Cq = np.squeeze(np.array(Cq),axis=2).T
        # print(Cp.shape)
        # print(np.stack((Cp,Cq),axis=0))
        print("CHECK THIS",np.mean(norm(Cp-Cq,axis=1)))
        tmp_st = time.time()
        R,t = GetTranform(Cp, Cq)
        transform_time_total += time.time() - tmp_st

        # P = R@P+t  # Newly, Should be this one
        P = R@Cp+t
        all_P = R@all_P+t
        newCost = np.sum( np.linalg.norm(P-Q , axis=0)**2  )
        
        # gc.collect()

        # P = R@P+t  # Newly, Should be this one
        # newCost = np.sum( np.linalg.norm(R@Cp+t-Cq , axis=0)**2  )
        error_list.append(newCost)

        if ( newCost < bestCost ):
            bestCost = newCost
            # bestP = copy(P)
            bestP = copy(all_P)
        print ( itr , " : ", bestCost )
        if ( newCost  < eps):
            doneFlag = True
        
        if ( itr > 30 and error_list[-5] - newCost < eps):
            print ( error_list[-5] - newCost )
            doneFlag = True

        itr = itr + 1
        if ( itr % 20 == 0 ):
            print ( "Iteration" , itr )

        del Cp, feature_p
        gc.collect()

    ed_time = time.time()

    print("Time Cost Total=", ed_time-st_time)
    print("Time Cost for filtering points =", np.sum(np.array(filter_time)))
    print("Time Cost for feature generation =", np.sum(np.array(feature_gen_time)))
    print("Time Cost for join =", join_time_total)
    print("Time for transform=", transform_time_total)
    print("Itr Total=", itr)
    print("Avg Selected Point=", np.mean(np.array(selected_points_num)))
    print("Best Cost is: ", bestCost)
    

    print ( P.shape )
    pc_fit = utils.convert_matrix_to_pc( np.expand_dims(bestP,axis=2) )
    pc_target = utils.convert_matrix_to_pc( np.expand_dims(Q,axis=2) )
    utils.view_pc([pc_fit, pc_target], None, ['b', 'r'], ['o', '^'])

    fig1 = plt.figure()
    plt.plot( range(len(error_list)) , error_list )

    pc_source = utils.convert_matrix_to_pc( np.expand_dims(ori_P,axis=2) )
    pc_target = utils.convert_matrix_to_pc( np.expand_dims(Q,axis=2) )
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.show()


if __name__ == '__main__':
    main()
