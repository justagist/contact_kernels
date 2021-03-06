"""
    "A kernel-based approach to learning contact distributions for robot manipulation tasks" [2017 Auton Robot]
            Oliver Kroemer, Simon Leischnig, Stefan Luettgen, Jan Peters
"""

'''
    -> Creates distributions from just contact points (and also with a combination of contact points and forces/normals.) 
    -> Computes bhattacharya kernels and bhattacharya distances for kernels
    -> Clusters contact distributions according to their 'similarity'
'''

import numpy as np
import math
from sklearn.cluster import KMeans

class KernelOperations:

    # def __init__(self, contact_pts):

    def bhattacharya_kernel(self, cnt_dist1, cnt_dist2):

        '''
            Calculates the Bhattacharya similarity kernel between two distributions. 0 < k <= 1 (k = 1 if distributions are identical)
            
            Arguments:
                cnt_dist1, cnt_dist2: ContactDistribution objects of same dimensions (3 or 6)

            Returns:
                k: kernel value

            Created using equations from [1] 

            [1]: http://math.uchicago.edu/~may/REU2013/REUPapers/Choe.pdf
        '''

        assert cnt_dist1.dim == cnt_dist2.dim

        mu_star = (0.5 * np.dot(np.linalg.inv(cnt_dist1.sigma), cnt_dist1.mu)) + (0.5 * np.dot(np.linalg.inv(cnt_dist2.sigma), cnt_dist2.mu))

        sigma_star = np.linalg.inv((0.5 * np.linalg.inv(cnt_dist1.sigma)) + (0.5 * np.linalg.inv(cnt_dist2.sigma)))

        exp_val = (-0.25 * np.dot(np.dot(cnt_dist1.mu, np.linalg.inv(cnt_dist1.sigma)), cnt_dist1.mu))- (0.25 * np.dot(np.dot(cnt_dist2.mu, np.linalg.inv(cnt_dist2.sigma)), cnt_dist2.mu))+ (0.5 * np.dot(np.dot(mu_star, sigma_star), mu_star))

        k = (np.linalg.det(sigma_star)**(0.5)) * (np.linalg.det(cnt_dist1.sigma)**(-0.25)) * (np.linalg.det(cnt_dist2.sigma)**(-0.25)) * math.exp(exp_val)

        return k

    def bhattacharya_distance(self, cnt_dist1, cnt_dist2):

        '''
            Calculate Bhattacharya distance between two kernels.

            Arguments:
                cnt_dist1, cnt_dist2: ContactDistribution objects of same dimensions (3 or 6)

            Returns:
                bat_distance: bhattacharya distance

        '''

        assert cnt_dist1.dim == cnt_dist2.dim

        mean1 = cnt_dist1.mu
        mean2 = cnt_dist2.mu
        cov1 = cnt_dist1.sigma
        cov2 = cnt_dist2.sigma

        cov_mean=(cov1+cov2)/2

        inv_cov_mean=np.linalg.pinv ( cov_mean )

        s = np.subtract(mean1,mean2)

        det1=np.linalg.norm(cov1)
        det2=np.linalg.norm(cov2)

        detmean=np.linalg.norm(cov_mean)    

        bat_distance=(0.125)*np.dot(np.dot(s.T,inv_cov_mean),s)+(0.5)*np.log(detmean/np.sqrt(det1*det2))

        return bat_distance

    def cluster_kernels(self, list_of_cnt_dists, num_clusters):

        '''
            Clusters different contact distributions into K clusters (according to the main paper [2]).

            Arguments:
                list_of_cnt_dists: the list of ContactDistribution objects
                num_clusters: Number of clusters

            Returns:
                ret_val: list of cluster IDs (representing the cluster to which the corresponding distribution belongs to)

            [2]: https://link.springer.com/article/10.1007%2Fs10514-017-9651-z

        '''

        num_samples = len(list_of_cnt_dists)

        assert num_clusters <= num_samples

        # Initialise D and K with zeros
        D = np.zeros([num_samples, num_samples]) # D: Diagonal Matrix; 
        K = np.zeros([num_samples, num_samples]) # K: Kernel Matrix
        # ----- populate D and K matrices
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):

                K[i,j] = self.bhattacharya_kernel(list_of_cnt_dists[i], list_of_cnt_dists[j])

                if i == j:

                    this_sample = list_of_cnt_dists[i]

                    total = 0
                    for sample in list_of_cnt_dists:
                        total += self.bhattacharya_kernel(this_sample, sample)

                    D[i,j] = total

        # ----- L: Normalised Laplacian
        L = np.eye(num_samples) - np.dot(np.linalg.inv(D),K)

        # ----- Find Eigen vectors for the Laplacian
        eigenValues, eigenVectors = np.linalg.eig(L)

        # ----- sort eigen vectors in the order of increasing eigen values
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        # ----- E: Eigen Matrix containing num_clusters Eigenvectors as columns (sorted as: lowest to highest eigenvalues)
        E = eigenVectors[:,:num_clusters]

        # ----- perform k-means on the rows of E
        ret_val = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(E)

        return ret_val



class ContactDistribution:

    """
        Class to create a contact distribution for the given contact points. 

    """
    def __init__(self, contact_pts, contact_adds = None, sigma_0 = 0):

        '''
            Arguments:
                contact_pts: list of positions of contact points in the interaction frame in the format [(x1,y1,z1),(x2,y2,z2),...]
                contact_adds: list of 3D forces/normals in the interaction frame in the format [(a1,b1,c1),(a2,b2,c2)...]

                sigma_0: (float or np.array() of size dxd, where d is the number of dimensions (6 if contact_adds present, 3 otherwise)) similarity between induvidual contacts. Can be adjusted to increase or decrease the importance of a dimension.
            
            Requirement:
                len(contact_pts) = len(contact_adds) if contact_adds is not None

        '''
        self._contact_adds = contact_adds
        self._contact_pts = contact_pts
        if contact_adds is None:
            self._contact_vec = np.asarray(contact_pts).T
            self.dim = 3

        else:
            assert len(contact_pts) == len(contact_adds)
            self._contact_vec = np.hstack([np.asarray(contact_pts),np.asarray(contact_adds)]).T
            self.dim = 6
            # self._contact_vec = 

        self._sigma_0 = np.dot(np.eye(self._contact_vec.shape[0]),sigma_0) 

        self.mu, self.sigma = self.create_distribution(self._contact_vec)


    def create_distribution(self, contact_pts):
        """
            Create contact distribution kernel
        """

        # assert contact_pts.shape[0] == 3 or contact_pts.shape[0] == 6

        mean = np.mean(contact_pts, axis=1)
        sigma = np.diag(np.diag(np.cov(contact_pts))) + self._sigma_0

        return mean, sigma

    def update_sigma_0(self, sigma_0):

        self._sigma_0 = sigma_0
        self.mu, self.sigma = self.create_distribution(self._contact_vec)

## ======================== ##
#         TEST CODE          #
## ======================== ##
if __name__ == '__main__':
    
    pts = [(1,2,3),(3.4,2.1,55.3),(6,94,76),(-23,34,1.3)]
    pts1 = [(3,2,300),(3.4,20.1,-55.3),(3.4,2.1,55.3),(1,-2,3)]
    pts2 = [(1,2.1,3),(33.4,21.1,5.3)]
    pts3 = [(2.3,4.3,4.4),(64.3,33.4,85.4)]
    pts4 = [(13,24,36),(36.4,72.1,565.3),(65,944,736),(-223,434,-1.3)]
    pts5 = [(-3,22,50),(13.4,420.1,-55.3),(3.4,-2.1,-55.3),(1,-2,-3)]    

    d1 = ContactDistribution(pts,pts1)
    d2 = ContactDistribution(pts2,pts3)
    d3 = ContactDistribution(pts4,pts5)

    ko = KernelOperations()

    # print ko.bhattacharya_kernel(d1,d2)
    # print ko.bhattacharya_distance(d1,d2)

    lst = [d1,d2,d1,d1,d1,d2,d2,d1,d2,d2,d3,d3,d3]

    print ko.cluster_kernels(lst,3)