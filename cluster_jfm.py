from cvxopt import matrix,spmatrix,sparse
import numpy as np
import math as math
from abstract_jfm import AbstractJFM


class ClusterJFM(AbstractJFM):
    """ Clustering Joint Feature Map for LatentSVDD."""

    num_classes = -1 # (scalar) number of classes 


    def __init__(self, X, classes, y=None):
        # the class also acts as indices therefore:
        # y >= 0!
        AbstractJFM.__init__(self, X, y)
        self.num_classes = classes      

    def argmax(self, sol, idx, add_loss=False):
        nd = self.feats
        mc = self.num_classes
        d = 0  # start of dimension in sol
        val = -10**10
        cls = -1 # best choice so far
        psi_idx = matrix(0.0, (nd*mc,1))

        for c in range(self.num_classes):

            psi = matrix(0.0, (nd*mc,1))
            psi[nd*c:nd*(c+1)] = self.X[:,idx]

            foo = 2.0 * sol.trans()*psi - psi.trans()*psi
            # the argmax of the above function
            # is equal to the argmax of the quadratic function
            # foo = + 2*foo - normPsi
            # since ||\Psi(x_i,z)|| = ||\phi(x_i)|| = y \forall z   
            if (np.single(foo)>np.single(val)):
                val = -sol.trans()*sol + foo
                cls = c
                psi_idx = matrix(sol, (nd*mc,1))
                psi_idx[nd*c:nd*(c+1)] = self.X[:,idx]
        return (val, cls, psi_idx)
     
    def get_num_dims(self):
        return self.feats*self.num_classes