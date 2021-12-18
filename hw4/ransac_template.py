#!/usr/bin/env python
from os import error
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random
import numpy as np
###YOUR IMPORTS HERE###

def FitModel(X):
    # print ( "Fit X.shape" , X.shape)
    miu = np.mean(X,axis=1).reshape( (3,1) )
    # miu =  np.mean(a=X , axis=1, keepdims=True)
    # print ( "miu shpae", miu.shape)
    Q = np.cov(X - miu)
    U, S, Vh = np.linalg.svd(Q)
    normalU = np.expand_dims(U[:,-1],axis=1)
    return normalU, miu

def ErrorFunc(p , normal_axis, mean_p):
    A = normal_axis[0]
    B = normal_axis[1]
    C = normal_axis[2]
    D = -np.sum( normal_axis.T@mean_p )
    if ( p.shape == (3,1) ):
        return (normal_axis.T@p+D)**2
    else:
        return np.sum( (p.T@normal_axis+D)**2 )
    

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    random.seed(498001)
    ###YOUR CODE HERE###
    # Show the input point cloud
    # fig1 = utils.view_pc([pc])

    #Fit a plane to the data using ransac
    # print ( random.sample( range(5) , 3) )
    P = utils.convert_pc_to_matrix(pc)
    max_itr = 5000
    N = P.shape[1]
    print ( N )
    delta = 1e-2
    M = 120
    e_best = 99999999
    best_para = []
    for itr in range(max_itr):
        picked_idx = random.sample(range(N),3)
        Psub = P[:,picked_idx] # Three points for a plane
        # normalU = np.cross( np.squeeze(Psub[:,1]-Psub[:,0]) , np.squeeze(Psub[:,2]-Psub[:,0]) ).T
        # centerU = np.mean(Psub,axis=1)
        normalU, centerU = FitModel(Psub)
        C = []
        rest_idx = [i for i in range(N) if i not in picked_idx]
        # print ( len(rest_idx) )
        Prest = P[:,rest_idx]
        # print ( "Prest.shape[1]= " , Prest.shape[1] )
        for i in range(Prest.shape[1]):
            p = Prest[:,i]
            if ( ErrorFunc(p , normalU, centerU) < delta ):
                C.append(p)
        if ( len(C) > M ):
            # print ( np.array(C).shape )
            for i in range(3):
                C.append(Psub[:,i])
            newPsub = np.squeeze( np.array(C).T )
            normalU, centerU = FitModel(newPsub)
            e_new = ErrorFunc(newPsub,normalU, centerU)
            if ( e_new < e_best ):
                e_best = e_new
                best_para = [normalU,centerU]
                print ( "itr=",itr , " e_new=",e_new , " len(C)=",len(C))
        if ( itr%1000 == 0 ):
            print ( "iteration: " , itr )
        # print ( Psub.shape )
        # pc_a = utils.convert_matrix_to_pc(Psub)
        # fig2 = utils.view_pc([pc_a])
        # print ( normalU.shape )
        # print ( Psub[:,0].shape )
        # fig2 = utils.draw_plane(fig2,normalU,Psub[:,0], color=(0,1,0,0.3))


    normalU = best_para[0]
    centerU = best_para[1]
    #Show the resulting point cloud
    #Draw the fitted plane
    # fig2 = utils.view_pc([pc],color='b')
    print ( "normalU" , normalU.shape )
    print ( "centerU" , centerU.shape )
    print ( normalU , centerU )
    D = -np.sum( normalU.T@centerU )
    theta_best = np.round([normalU[0,0],normalU[1,0],normalU[2,0],D],3)
    print ( "Plane: 0 = ", theta_best[0], "*x + " , theta_best[1], "*y + " , theta_best[2], "*z + ", theta_best[3])

    inliers_idx = []
    outliers_idx = []
    for i in range(N):
        p = P[:,i]
        if ( ErrorFunc(p , normalU, centerU) < delta ):
            inliers_idx.append(i)
        else:
            outliers_idx.append(i)
    inliers = P[:,inliers_idx]
    outliers = P[:,outliers_idx]
    
    fig1 = utils.view_pc([ utils.convert_matrix_to_pc(inliers) ] , color='r')
    fig1 = utils.view_pc([ utils.convert_matrix_to_pc(outliers) ],fig=fig1,color='b')
    fig1 = utils.draw_plane(fig1,normalU,centerU, color=(0,1,0,0.3))
    
    


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
