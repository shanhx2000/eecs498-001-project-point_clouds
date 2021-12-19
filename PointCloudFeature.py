#!/usr/bin/env python
import numpy as np
from numpy.lib.utils import source
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from math import floor, pi

class point_and_feature:

    def __init__(self, point, feature, normal, curvature) -> None:
        self.point = point
        self.feature = feature
        self.normal = normal
        self.curvature = curvature

def set_limit(v, minv, maxv):
    if v < minv:
        return minv+1e-7
    elif v > maxv:
        return maxv-1e-7
    return v

class PointCloudFeature:
    
    def __init__(self, 
                source, 
                distance_for_patch=1e-2, 
                curvature_for_patch=0.9, 
                verbose=False,
                distance_for_patch_n_k=9e-3,
                ) -> None:
        self.source = np.array(source)
        self.distance_for_patch = distance_for_patch
        self.distance_for_patch_n_k = distance_for_patch_n_k
        self.curvature_for_patch = curvature_for_patch
        self.p_f_list = []
        self.curvature_list = []
        self.normal_list = []
        self.N = source.shape[1]

        self.verbose = verbose
        if verbose:
            self.patch_size_list = []
            print("PCF source shape=", self.source.shape)
    
    def build_features(self):
        assert self.p_f_list == []
        N = self.source.shape[1]
        if self.verbose:
            print("Start building up PCF")

        for i in range(N):
            p = self.source[:,i].reshape( (3,1) )
            patch = self.select_patch(p, distance=self.distance_for_patch_n_k)
            self.gen_n_k(point=p, patch=patch)

        if self.verbose:
            fig1 = plt.figure()
            plt.hist(self.curvature_list, bins='auto')
            plt.title("Curvature Distribution")
            plt.show()

            fig2 = plt.figure()
            plt.hist(self.patch_size_list, bins='auto')
            plt.title("Patch Size Distribution")
            plt.show()

        for i in range(N):
            p = self.load(i)
            if p is None:
                continue
            
            patch_idx = self.select_patch_index(p, self.distance_for_patch)
            feature, normal, curvature = self.gen_feature(point_idx=i, patch_idx=patch_idx)
            
            if feature is not None:
                self.p_f_list.append(point_and_feature(p, feature, normal, curvature))
            
            if self.verbose:
                if i%40 == 0:
                    print("feature for ", i, " is: ", feature, "\n")

        if self.verbose:
            print("Obtain #feature=", len(self.p_f_list))

        # assert False
        return len(self.p_f_list)

    def select_patch_index(self, point, distance):
        return norm(self.source - point, axis=0) < distance

    def select_patch(self, point, distance):
        # print(  "source=", self.source.shape, "\n",
        #         "point=", point.shape, "\n",
        #         "self.source - point=", (self.source - point).shape, "\n",
        #         "norm(self.source - point, axis=0)=", norm(self.source - point, axis=0).shape, "\n",
        #     )
        return self.source[:, self.select_patch_index(point, distance=distance)]

    def compute_curvature(self, S):
        k = S[0] / (S[0]+S[1]+S[2])
        return k

    def gen_n_k(self, point, patch):
        U, S, Vh = svd(patch@patch.T)
        # print("U=", U.shape, "\n",
        #         "S=", S.shape, "\n",
        #         "Vh=", Vh.shape, "\n",)  # U=(3,3), S=(3,), Vh=(3,3)
        if self.verbose:
            self.patch_size_list.append(patch.shape[1])

        normal = U[:,0]
        self.normal_list.append(normal)

        curvature = self.compute_curvature(S)
        self.curvature_list.append(curvature)

        if self.verbose:
            self.patch_size_list.append(patch.shape[1])
        pass        
    
    def calc_signature(self, alpha, phi, theta, signature_set):
        alpha_offset = -0.07
        alpha_range = 0.14
        phi_offset = 0.0
        phi_range = 0.6
        theta_offset = -pi/10.
        theta_range = pi/10.*2.
        bins = 4  # 0~bins-1

        alpha = set_limit(alpha, alpha_offset, alpha_offset+alpha_range)
        phi = set_limit(phi, phi_offset, phi_offset+phi_range)
        theta = set_limit(theta, theta_offset, theta_offset+theta_range)

        s1 = floor(bins*(alpha-alpha_offset)/alpha_range)
        s2 = floor(bins*(phi-phi_offset)/phi_range)
        s3 = floor(bins*(theta-theta_offset)/theta_range)
        signature_set[s1+s2*bins+s3*bins*bins] += 1
        return signature_set

    def gen_feature(self, point_idx, patch_idx):
        point = self.source[:, point_idx]
        patch = self.source[:, patch_idx]

        U, S, Vh = svd(patch@patch.T)
        # print("U=", U.shape, "\n",
        #         "S=", S.shape, "\n",
        #         "Vh=", Vh.shape, "\n",)  # U=(3,3), S=(3,), Vh=(3,3)
        # normal = U[:,0]
        # curvature = self.compute_curvature(S)

        curvature = self.curvature_list[point_idx]

        if curvature < self.curvature_for_patch:
            return None, None, None
        
        tmp_feature = 0
        tmp_normal = 0

        signature_set = np.zeros(4**3,)
        # print(signature_set.shape)
        # assert False

        # print("patch", patch_idx.shape)


        # ns = self.normal_list[point_idx]
        # ps = point

        pt_idx_list = np.arange(self.N)[patch_idx]
        for pt_idx in pt_idx_list:
            for pt_idx2 in pt_idx_list:
                if pt_idx == pt_idx2:
                    continue
                pt = self.source[:, pt_idx].reshape(3,)
                nt = self.normal_list[pt_idx].reshape(3,)
                ps = self.source[:, pt_idx2].reshape(3,)
                ns = self.normal_list[pt_idx2].reshape(3,)
                
                d = norm(pt-ps)
                if d < 1e-7:  # Itself
                    continue
                u = ns
                v = np.cross(u, (pt-ps)/d)
                w = np.cross(u, v)

                alpha = np.dot(v, nt)
                phi = np.dot(u, (pt-ps)/d)
                theta = np.arctan2(np.dot(w, nt), np.dot(u, nt))

                # print(
                #     "v=", v, "\n",
                #     "ns=", ns, "\n",
                #     "alpha=", alpha, "\n",
                #     "phi=", phi, "\n",
                #     "theta=", theta, "\n",
                # )

                signature_set = self.calc_signature(alpha, phi, theta, signature_set)
        
        # print(len(pt_idx_list))
        # print(signature_set)
        # assert False

        return signature_set, self.normal_list[point_idx], curvature


    # Load feature tuple at index
    # If not a good point, return None
    def load(self, index):
        if self.good_feature(index):
            return self.source[:,index].reshape( (3,1) )
        else:
            return None
    
    # # PCF.find return the index of feature, which is nearest to the input feature. 
    # def find(self, target_feature):
    #     pass

    # Check if this data point a good one for feature calculation
    def good_feature(self, index):
        if self.curvature_list[index] < self.curvature_for_patch:
            return False
        return True

def main():
    tmp = PointCloudFeature()
    pass


if __name__ == '__main__':
    main()
