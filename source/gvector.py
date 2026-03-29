import numpy as np
import scipy
from scipy import linalg
from scipy import special
import copy


# Reciprocal grid point


def mm_to_nn(i,n):
    if i > n // 2:
        i=i-n
    return(i)


class G_vector(object):
    def __init__(self, rec_latt, wf_cutoff, grid_point):
    
        
        self.grid_point    = grid_point
    
        self.n_gw          = 0
        self.n_point       = np.prod(grid_point)
        self.g_vector_mask = np.zeros(grid_point)
        self.g2_vector     = np.zeros(grid_point)
        self.g2_vector[:,:,:]     = -1
        self.g1_vector     = np.zeros((grid_point[0],grid_point[1],grid_point[2],3))
        
        
        
        for i in range(grid_point[0]):
            for j in range(grid_point[1]):
                for k in range(grid_point[2]):
                    ii,jj,kk=mm_to_nn(i, grid_point[0]), mm_to_nn(j,grid_point[1]), mm_to_nn(k, grid_point[2])
                    temp=ii*rec_latt[0] + jj*rec_latt[1] + kk*rec_latt[2]
                    temp1=np.sum(temp**2)                    
                    if temp1 <= 8. * wf_cutoff:
                        self.g_vector_mask[i,j,k] = 1
                        self.n_gw += 1
                        self.g2_vector[i,j,k] = temp1
                        self.g1_vector[i,j,k] = temp
                        
    
        self.g_vector_mask =  self.g_vector_mask == 1

        # store the wavefunction g-vectors


        self.gx2_vector     = np.zeros(grid_point, dtype = np.float64)
        self.gx1_vector     = np.zeros((grid_point[0],grid_point[1],grid_point[2],3),dtype = np.float64)

        self.gx_vector_mask = np.logical_and( self.g2_vector <= 2.* wf_cutoff,  self.g2_vector >= 0.)

        self.n_gxw          = np.sum(self.gx_vector_mask)

        self.gx1_vector[self.gx_vector_mask]     =  self.g1_vector[self.gx_vector_mask]          
        self.gx2_vector[self.gx_vector_mask]     =  self.g2_vector[self.gx_vector_mask]

        self.g2_vector[self.g2_vector == -1 ] = 0
        # print("g2_min", np.min(self.g2_vector[self.g_vector_mask>0.]))
    

    def cal_op_coul(self, beta = 0.0 ):
        self.op_coul  = np.zeros(self.grid_point, dtype = np.float64)
        self.op_coul[self.g_vector_mask]  = self.g2_vector[self.g_vector_mask]

        self.op_coul[0,0,0] = 1.0  # avoid divied by zero
        self.op_coul[self.g_vector_mask]= 4 * np.pi / self.op_coul[self.g_vector_mask]
        self.op_coul[0,0,0] = 0.0

        if beta > 0.0 :
            self.op_coul[self.g_vector_mask] = self.op_coul[self.g_vector_mask] *(1.0 - \
                                                np.exp(-self.g2_vector[self.g_vector_mask]/(4*beta**2)))
            # when g->0 , 4pi(1.0-exp(-|g|^2/4/beta^2))/|g|^2 -> pi/beta^2  (tailor expansion)
            # which is same to the method in quantum espresso
            self.op_coul[0,0,0] = np.pi / beta**2
