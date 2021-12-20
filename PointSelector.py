from sklearn.cluster import KMeans

def no_select(P1, P2):
    return P1, P2

def Kmeans_select(P1, P2, ratio = 0.1):
    # return no_select(P1, P2)
    N = P1.shape[-1]
    kmeans1 = KMeans(n_clusters=(int)(ratio*N), max_iter=300, random_state=0).fit(P1.T)
    P1_ret = kmeans1.cluster_centers_.T #P1[:,kmeans1.labels_]
    kmeans2 = KMeans(n_clusters=(int)(ratio*N), max_iter=300, random_state=0).fit(P2.T)
    P2_ret = kmeans2.cluster_centers_.T  #P2[:,kmeans2.labels_]
    return P1_ret, P2_ret