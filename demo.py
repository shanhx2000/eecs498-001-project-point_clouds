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
import time
from PointSelector import Kmeans_select
from copy import copy
from sparsePC import sparse_PC
from randomPC import random_PC

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

def main():

    # tunable params
    dataset_input = input('Please select a dataset from cat, horse, wolf:\n')
    if dataset_input == 'cat' or dataset_input == 'horse' or dataset_input == 'wolf':
        dataset = dataset_input
    else:
        print("Incorrect Input, Run with default set (cat).")
        dataset = 'cat' # Other Set includes: horse, wolf ; 
    
    if dataset == 'cat':
        ratio = 0.1
        dist = 10
    elif dataset == 'horse':
        ratio = 0.3
        dist = 10
    elif dataset == 'wolf':
        ratio = 0.3
        dist = 10
    # tunable params end

    bins = 5
    bins_input = eval(input('Please select bins from 4, 5, 6:\n'))
    if (bins_input < 4 or bins_input > 6):
        print("Incorrect Input, Run with default bins (5).")
    else:
        bins = (int)(bins_input)

    print("Start Loading Data")

    # Import Data
    with open('data_pcd/ism_train_{}_source.npy'.format(dataset),'rb') as f:
        PC1 = np.load(f)
    with open('data_pcd/ism_train_{}_target.npy'.format(dataset),'rb') as f:
        PC2 = np.load(f)
    P = PC1.T
    Q = PC2.T
    all_Q = copy(Q)

    # Init
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
    ori_all_P = copy(P)
    tmp_st = time.time()
    P, Q = Kmeans_select(P, Q, ratio=ratio)
    filter_time.append(time.time()-tmp_st)


    print("Build Q=", Q.shape)
    tmp_st = time.time()
    feature_q = PCF(Q, distance_for_patch = dist, bins=bins)
    fq = feature_q.build_features()
    feature_gen_time.append(time.time()-tmp_st)
    
    print("Start ICP!")

    while ( not doneFlag and itr < 100 ):
        Cp = []
        Cq = []
        
        tmp_st = time.time()
        P_filterred, Q_filterred = P, Q

        selected_points_num.append(P_filterred.shape[-1])
        filter_time.append(time.time()-tmp_st)
        # print("Generate ", P_filterred.shape[-1], " Points for P")

        tmp_st = time.time()
        feature_p = PCF(P_filterred, verbose=False, distance_for_patch = dist, bins=bins)
        fp = feature_p.build_features()
        feature_gen_time.append(time.time()-tmp_st)
        
        assert feature_p != []

        tmp_st = time.time()
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
        tmp_st = time.time()
        R,t = GetTranform(Cp, Cq)
        transform_time_total += time.time() - tmp_st

        P = R@Cp+t
        all_P = R@all_P+t
        newCost = np.sum( np.linalg.norm(P-Q , axis=0)**2  )
        error_list.append(newCost)

        if ( newCost < bestCost ):
            bestCost = newCost
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
    pc_target = utils.convert_matrix_to_pc( np.expand_dims(all_Q,axis=2) )
    utils.view_pc([pc_fit, pc_target], None, ['b', 'r'], ['o', '^'])
    # plt.savefig('result/{}_ratio_{}_dist_{}.png'.format(dataset, ratio, dist),bbox_inches='tight')
    # plt.savefig('result/{}_without_k_means.png'.format(dataset),bbox_inches='tight')

    fig1 = plt.figure()
    plt.plot( range(len(error_list)) , error_list )
    plt.title("Loss v.s. Iterations")
    plt.show()

    # pc_source = utils.convert_matrix_to_pc( np.expand_dims(ori_all_P,axis=2) )
    # pc_target = utils.convert_matrix_to_pc( np.expand_dims(all_Q,axis=2) )
    # utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    # plt.savefig('result/{}_ori.png'.format(dataset),bbox_inches='tight')

if __name__ == '__main__':
    main()
