import numpy as np

def randomPC(PC1, PC2, N):
    """
    @params PC1: the source PC with size [#, 3]
    @params PC2: the target PC with size [#, 3]
    @params N: the number of points you want after filtering

    return P, Q with size [3, N]
    """
    PC1_rand_index = np.random.choice(PC1.shape[0], N, replace=False)
    PC2_rand_index = np.random.choice(PC2.shape[0], N, replace=False)
    tmp_P1 = PC1[PC1_rand_index, :]
    tmp_P2 = PC2[PC2_rand_index, :]
    P1 = np.expand_dims(tmp_PC1, axis = 2)
    P2 = np.expand_dims(tmp_PC2, axis = 2)
    P = utils.convert_pc_to_matrix(P1)
    Q = utils.convert_pc_to_matrix(P2)
    return P, Q

