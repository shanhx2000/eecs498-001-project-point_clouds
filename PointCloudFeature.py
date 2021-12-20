#!/usr/bin/env python
from re import T
import numpy as np
from numpy.lib.utils import source
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from math import floor, pi

from numpy.ma.core import set_fill_value

class point_and_feature:

    def __init__(self, point, feature) -> None:
        self.point = point
        self.feature = feature

def set_limit(v, minv, maxv):
    if v < minv:
        return minv+1e-7
    elif v > maxv:
        return maxv-1e-7
    return v

class PointCloudFeature:
    
    # def __init__(self, 
    #             to_handle,
    #             source, 
    #             distance_for_patch=1.5*1e-2, 
    #             curvature_for_patch=7e-7, 
    #             verbose=False,
    #             distance_for_patch_n_k=1.5*1e-2,
    #             ) -> None:
    # def __init__(self, 
    #             to_handle,
    #             source, 
    #             distance_for_patch=3e-2, 
    #             curvature_for_patch=7e-7, 
    #             verbose=False,
    #             distance_for_patch_n_k=2e-2,
    #             ) -> None:
    def __init__(self, 
                source, 
                distance_for_patch=10, 
                curvature_for_patch=7e-7, 
                verbose=False,
                distance_for_patch_n_k=3e-2,
                ) -> None:

        self.source = np.array(source)
        self.distance_for_patch = distance_for_patch
        self.distance_for_patch_n_k = distance_for_patch_n_k
        self.curvature_for_patch = curvature_for_patch
        self.p_f_list = []
        self.curvature_list = []
        self.normal_list = []
        self.dist_mat = self.get_dist_mat(self.source)

        N = source.shape[1]
        for i in range(N):
            self.curvature_list.append(None)
            self.normal_list.append(None)

        self.patch_size = []
        self.N = source.shape[1]
        # self.output_points = []

        self.bins = 5

        self.verbose = verbose
        if verbose:
            self.patch_size_list = []
            print("PCF source shape=", self.source.shape)
    
    def get_dist_mat(self, x):
        x = x.T
        print(x.shape)
        s1 = np.sum(x**2, axis=1, keepdims=True)
        s2 = np.sum(x**2, axis=1, keepdims=True)
        dist = (s1 + s2.T - 2.0 * x@x.T)
        dist[dist < 1e-7] = 0
        return dist**0.5

    def build_features(self):
        
        
        assert self.p_f_list == []
        N = self.source.shape[1]

        for i in range(N):
            p = self.source[:,i].reshape( (3,1) )
            patch = self.select_patch(point=p, distance=self.distance_for_patch)
            patch_idx = np.squeeze(np.argwhere(self.dist_mat[i,:] < self.distance_for_patch))
            # assert np.all(patch == self.source[:,patch_idx])

            self.patch_size.append(len(patch))
            _, S, Vh = svd(patch@patch.T)
            min_idx = np.argmin(S)
            normal = Vh[min_idx,:]
            self.normal_list[i] = normal

        self.normal_list = np.array(self.normal_list)
        print("Finish compute u")
        
        features = []
        for i in range(N):
            p = self.source[:,i].reshape( (3,1) )
            patch_idx = self.select_patch_index(point=p, distance=self.distance_for_patch)
            feature, _, _ = self.gen_feature(point_idx=i, patch_idx=patch_idx)
            self.p_f_list.append(point_and_feature(p, feature))
            features.append(feature)

        print("Avg Patch Size=", np.mean(np.array(self.patch_size)), "Max Patch Size=", np.max(np.array(self.patch_size)))
        
        features = np.array(features)
        return features

    def select_patch_index(self, point, distance):
        return norm(self.source - point, axis=0) < distance

    def select_patch(self, point, distance):
        return self.source[:, self.select_patch_index(point, distance=distance)]

    def compute_curvature(self, S):
        # k = S[0] / (S[0]+S[1]+S[2])
        k = S[2] / (S[0]+S[1]+S[2])
        return k

    def gen_n_k(self, point_id, patch):
        _, S, Vh = svd(patch@patch.T)
        if self.verbose:
            self.patch_size_list.append(patch.shape[1])

        min_idx = np.argmin(S)
        normal = Vh[min_idx,:]
        self.normal_list[point_id] = normal
        curvature = self.compute_curvature(S)
        self.curvature_list[point_id] = curvature
    
    def calc_signature(self, alpha, phi, theta, signature_set):
        alpha_offset = -0.07
        alpha_range = 0.14
        phi_offset = 0.0
        phi_range = 0.6
        theta_offset = -pi/10.
        theta_range = pi/10.*2.
        bins = self.bins  # 0~bins-1

        alpha = set_limit(alpha, alpha_offset, alpha_offset+alpha_range)
        phi = set_limit(phi, phi_offset, phi_offset+phi_range)
        theta = set_limit(theta, theta_offset, theta_offset+theta_range)
        # print(alpha, phi, theta)
        s1 = floor(bins*(alpha-alpha_offset)/alpha_range)
        s2 = floor(bins*(phi-phi_offset)/phi_range)
        s3 = floor(bins*(theta-theta_offset)/theta_range)
        signature_set[s1+s2*bins+s3*bins*bins] += 1
        return signature_set

    def gen_feature(self, point_idx, patch_idx):
        point = self.source[:, point_idx]
        # patch = self.source[:, patch_idx]

        patch = self.source[:, patch_idx].T
        patch_size = patch.shape[0]
       

        patch_u = self.normal_list[patch_idx, :]
        patch_dist = (self.dist_mat[patch_idx])[:, patch_idx]
        # print(patch_dist.shape, patch_size)
        assert patch_dist.shape == (patch_size, patch_size)

        u = np.repeat(patch_u[:, None], patch_size, axis=1)
        assert u.shape == (patch_size, patch_size, 3)

        pq_divide_d = (patch[:, None] - patch[None]) / (patch_dist[..., None] + 1e-7)
        pq_divide_d[np.eye(patch_size, dtype=bool)] = 0
        assert pq_divide_d.shape == (patch_size, patch_size, 3)

        v = np.cross(u, pq_divide_d)
        w = np.cross(u, v)

        alpha = np.sum(v * patch_u[:, None], axis=2)
        phi = np.sum(u * pq_divide_d, axis=2)
        theta = np.arctan2( np.sum(w * patch_u[:, None], axis=2), np.sum(u * patch_u[:, None], axis=2))

        idx = (np.tri(patch_size) - np.eye(patch_size)).astype(bool)
        alphas = alpha[idx]
        phis = phi[idx]
        thetas = theta[idx]

        # Old version
        # signature_set = np.zeros(self.bins**3,)
        # pt_idx_list = np.arange(self.N)[patch_idx]
        
        # alphas = []
        # phis = []
        # thetas = []

        # for pt_idx in pt_idx_list:
        #     for pt_idx2 in pt_idx_list:
        #         if pt_idx == pt_idx2:
        #             continue

        #         ps = self.source[:, pt_idx].reshape(3,)
        #         ns = self.normal_list[pt_idx].reshape(3,)
        #         pt = self.source[:, pt_idx2].reshape(3,)
        #         nt = self.normal_list[pt_idx2].reshape(3,)
                
        #         d = norm(pt-ps)
        #         if d < 1e-7:  # Itself
        #             continue

        #         u = ns
        #         v = np.cross(u, (pt-ps)/(d+1e-8))
        #         w = np.cross(u, v)

        #         alpha = np.dot(v, nt)
        #         phi = np.dot(u, (pt-ps)/(d+1e-8))
        #         theta = np.arctan2(np.dot(w, nt), np.dot(u, nt))

        #         alphas.append(alpha)
        #         phis.append(phi)
        #         thetas.append(theta)
        #         # signature_set = self.calc_signature(alpha, phi, theta, signature_set)
        
        # alphas = np.array(alphas)
        # phis = np.array(phis)
        # thetas = np.array(thetas)

        assert self.bins == 5
        signature_set, _ = np.histogramdd(
            np.stack([alphas, phis, thetas], axis=1), 
            bins=[
                [-1.0,-0.05, -1e-3, 1e-3, 0.05,1.0],
                [0.0, 0.05, 0.15, 0.3, 0.6, 1.0],
                [-pi, -pi/10., -pi/20., pi/20., pi/10., pi]
            ]
            )
        # signature_set, _ = np.histogramdd(
        #     np.stack([alphas, phis, thetas], axis=1), 
        #     bins=self.bins
        #     )

        signature_set = signature_set.flatten()
        # print(signature_set.shape)
        # assert False
        return signature_set, None, None  # self.normal_list[point_idx] curvature

    # Load feature tuple at index
    # If not a good point, return None
    def load(self, index):
        if self.good_feature(index):
            return self.source[:,index].reshape( (3,1) )
        else:
            return None
    
    # Check if this data point a good one for feature calculation
    def good_feature(self, index):
        # if self.curvature_list[index] < self.curvature_for_patch:
        #     return False
        return True

def main():
    tmp = PointCloudFeature()
    pass


if __name__ == '__main__':
    main()
