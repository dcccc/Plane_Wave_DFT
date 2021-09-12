import numpy as np
import scipy
from scipy import linalg
from scipy import special
	
	
# init wavefunction

def orth1(psi):
    psi=np.dot(linalg.inv(linalg.sqrtm(np.dot(np.conj(psi),psi.T))).T, psi)
    return(psi)
    

def get_init_wf(n_state_all, n_gxw, gx_vector_mask, grid_point): 

    
    psi_g_1d = [np.random.rand(n_gxw) + np.random.rand(n_gxw) * 1.0j for i in range(n_state_all) ]
    
    
    psi_g_1d = np.array(psi_g_1d)
    
    psi_g_1d = orth1(psi_g_1d)
    
    psi_g_3d = []
    
    for i in range(n_state_all):
    
        psi_g_tmp  = np.zeros(grid_point, dtype = np.complex)
    
        psi_g_tmp[gx_vector_mask] = psi_g_1d[i]
        psi_g_3d.append(psi_g_tmp)                    
                                 
    psi_g_3d = np.array(psi_g_3d)
    
    return(psi_g_3d)
	
	
#  electron density

def cal_rhoe(psi_g_3d, vol, state_occupy_list, grid_point): 
    
    n_point   = np.prod(grid_point)
    rho_r_all = np.zeros(grid_point)
  

    for i, ratio in enumerate(state_occupy_list):
        
        if ratio > 0.0:

            psi_r = np.fft.ifftn(psi_g_3d[i])
            psi_r = (n_point / vol)**0.5 * psi_r
            
            rho_r = np.real(psi_r * np.conj(psi_r))
        
            rho_r = rho_r / np.sum(rho_r) * 2. / vol * n_point * ratio
            
            rho_r_all = rho_r + rho_r_all
        
    rho_r_all = rho_r_all / np.sum(rho_r_all) * np.sum(state_occupy_list) * 2. / vol * n_point 

    return(rho_r_all)

	
	
