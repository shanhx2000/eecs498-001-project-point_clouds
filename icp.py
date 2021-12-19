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
from pcloader import load_pcl
import time
import pcl

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
    pc_source = utils.load_pc('hw4/cloud_icp_source.csv')
    pc_target = utils.load_pc('hw4/cloud_icp_target3.csv') # Change this to load in a different target
    P = utils.convert_pc_to_matrix(pc_source)
    Q = utils.convert_pc_to_matrix(pc_target)
    

    # data_dir = "./data/tutorials/template_alignment"
    # filename_source = "person.pcd"
    # filename_target = "object_template_0.pcd"
    # source_data = load_pcl(filename=(data_dir+filename_source))
    # target_data = load_pcl(filename=(data_dir+filename_target))
    # P = np.matrix(source_data.T)
    # Q = np.matrix(target_data.T)

    # print ( P.shape )
    # fig1 = plt.figure()
    # pc_fit = utils.convert_matrix_to_pc( np.expand_dims(P,axis=2) )
    # utils.view_pc([pc_fit, pc_target], fig1, ['b', 'r'], ['o', '^'])
    # fig2 = plt.figure()
    # pc_fit = utils.convert_matrix_to_pc( np.expand_dims(Q,axis=2) )
    # utils.view_pc([pc_fit, pc_target], fig2, ['b', 'r'], ['o', '^'])

    # plt.axis([-0.15, 0.15, -0.15, 0.15])
    # plt.show()
    
    # print ( P.shape )
    # assert False

    st_time = time.time()
    
    
    doneFlag = False
    bestCost = 99999999
    eps = 1e-3 # 1e-2
    itr = 0
    error_list = []
    while ( not doneFlag and itr < 100 ):
        Cp = []
        Cq = []
        feature_p = PCF(P, verbose=False)
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
        if ( newCost  < eps):
            doneFlag = True
        
        if ( itr > 10 and error_list[-5] - newCost < eps):
            print ( error_list[-5] - newCost )
            doneFlag = True

        itr = itr + 1
        if ( itr % 20 == 0 ):
            print ( "Iteration" , itr )

    ed_time = time.time()

    print("Time Cost=", ed_time-st_time)

    print ( P.shape )
    pc_fit = utils.convert_matrix_to_pc( np.expand_dims(P,axis=2) )
    utils.view_pc([pc_fit, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])

    fig1 = plt.figure()
    plt.plot( range(len(error_list)) , error_list )
    plt.show()


if __name__ == '__main__':
    main()
