#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    X = utils.convert_pc_to_matrix(pc)
    miu = np.mean(X,axis=1)
    X = X - miu
    Q = np.cov(X)
    U, S, Vh = np.linalg.svd(Q)
    X_new = Vh @ X
    print ( "V.T=", np.round(Vh,3))

    #Show the resulting point cloud
    pc_a = utils.convert_matrix_to_pc(X_new)
    fig = utils.view_pc([pc_a] , fig, color='r')
    plt.savefig( "IMPL2_PCA_a.png" ) 


    #Rotate the points to align with the XY plane AND eliminate the noise
    fig1 = utils.view_pc([pc_a], color='r')
    s = np.diag(S**2)
    threshold = 1e-3
    idx_to_remove = []
    idx_keep = []
    for i in range ( s.shape[0] ):
        if ( s[i,i] < threshold ):
            idx_to_remove.append(i)
        else:
            idx_keep.append(i)
    # Vs = Vh[:,idx_keep]
    Vs = Vh
    Vs[:,idx_to_remove] = np.zeros_like(Vs[:,idx_to_remove])
    print ( "Vs.T", np.round(Vs.T,3) )
    X_b = Vs.T @ X
    pc_b = utils.convert_matrix_to_pc( X_b )
    fig1 = utils.view_pc([pc_b], fig1, color='g')
    plt.savefig( "IMPL2_PCA_b.png" )

    # Show the resulting point cloud

    ###YOUR CODE HERE###
    fig2 = utils.view_pc([pc])
    normalU = np.expand_dims(U[:,-1],axis=1)
    print ( normalU.shape )
    print ( miu.shape )
    print ( normalU )
    fig2 = utils.draw_plane(fig2,normalU,miu, color=(0,1,0,0.3))
    plt.savefig( "IMPL2_PCA_c.png" )
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
