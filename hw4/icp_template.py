#!/usr/bin/env python
from os import error
from numpy.core.defchararray import decode
from numpy.lib.twodim_base import eye
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

###YOUR IMPORTS HERE###
def GetTranform( Cp, Cq ):
    # print ( Cp.shape , Cq.shape )
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
    # R = R.T
    t = q - R@p
    return R,t

def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target

    P = utils.convert_pc_to_matrix(pc_source)
    Q = utils.convert_pc_to_matrix(pc_target)
    print ( P.shape )
    doneFlag = False
    bestCost = 99999999
    eps = 1e-3 # 1e-2
    itr = 0
    error_list = []
    while ( not doneFlag and itr < 100 ):
        Cp = []
        Cq = []
        for i in range(P.shape[1]):
            p = np.array(P[:,i]).reshape( (3,1) )
            # q_idx = -1
            # q_s = 999
            # for j in range(Q.shape[1]):
            #     tmp = np.linalg.norm(Q[:,j]-P[:,i])
            #     if ( tmp < q_s ):
            #         q_s = tmp
            #         q_idx = j
            q_idx = np.argmin( np.linalg.norm(Q-p,axis=0) )
            # print ( q_idx )
            # print ( np.argmin( np.linalg.norm(Q-p,axis=0) ) )
            # assert ( q_idx == np.argmin( np.linalg.norm(Q-p,axis=0) ))
            q = np.array(Q[:,q_idx]).reshape( (3,1) )
            Cp.append( p )
            Cq.append( q )
        Cp = np.squeeze(np.array(Cp),axis=2).T
        Cq = np.squeeze(np.array(Cq),axis=2).T
        R,t = GetTranform( Cp, Cq )
        newCost = np.sum( np.linalg.norm(R@Cp+t-Cq , axis=0)**2  )
        error_list.append(newCost)
        if ( newCost < bestCost ):
            bestCost = newCost
            print ( itr , " : ", bestCost )
        if ( newCost  < eps):
            doneFlag = True
        
        if ( itr > 6 and error_list[-5] - newCost < eps):
            print ( error_list[-5] - newCost )
            doneFlag = True
        P = R@Cp+t
        itr = itr + 1
        if ( itr % 20 == 0 ):
            print ( "Iteration" , itr )

    print ( P.shape )
    pc_fit = utils.convert_matrix_to_pc( np.expand_dims(P,axis=2) )
    utils.view_pc([pc_fit, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])

    fig1 = plt.figure()
    plt.plot( range(len(error_list)) , error_list )
    plt.show()
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
