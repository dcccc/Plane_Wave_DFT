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
    
        psi_g_tmp  = np.zeros(grid_point, dtype = np.complex128)
    
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


def nabla_dot(hx,hy,hz, g1_vector):
    hx_g = np.fft.fftn(hx)
    hy_g = np.fft.fftn(hy)
    hz_g = np.fft.fftn(hz)
    h_div = hx_g * g1_vector[:,:,:,0] + hy_g * g1_vector[:,:,:,1] + hz_g * g1_vector[:,:,:,2]
    h_div_r = np.real(np.fft.ifftn(h_div*1.j))
    return(h_div_r)


def cal_knetic_density(psi_g_3d, g1_vector, vol, state_occupy_list): 
    "tau"
    n_point   = np.prod(psi_g_3d[0].shape)
    scale     = n_point / vol**0.5
    kden      = np.zeros(psi_g_3d[0].shape)
    

    for i, ratio in enumerate(state_occupy_list):        
        if ratio > 0.0:
            psi_g = psi_g_3d[i]
            dx = psi_g * g1_vector[:,:,:,0] * 1.j
            dy = psi_g * g1_vector[:,:,:,1] * 1.j
            dz = psi_g * g1_vector[:,:,:,2] * 1.j
            dx_r = np.fft.ifftn(dx)*scale
            dy_r = np.fft.ifftn(dy)*scale
            dz_r = np.fft.ifftn(dz)*scale

            kden = kden + np.real(dx_r * np.conj(dx_r) + dy_r * np.conj(dy_r) + dz_r * np.conj(dz_r))    

    return(kden)


def cal_rhoe_grad(rho_all, g1_vector): 
    
    rho_g = np.fft.fftn(rho_all)

    rho_grad_r_x = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,0] * 1.j))
    rho_grad_r_y = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,1] * 1.j))
    rho_grad_r_z = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,2] * 1.j))


    # rho_lap_r_x = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,0]**2 * -1.))
    # rho_lap_r_y = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,1]**2 * -1.))
    # rho_lap_r_z = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,2]**2 * -1.))

    # rho_lap_r = rho_lap_r_x + rho_lap_r_y + rho_lap_r_z
    return(rho_grad_r_x, rho_grad_r_y, rho_grad_r_z)

def cal_lapl(rho_all, g1_vector): 
    
    rho_g = np.fft.fftn(rho_all)
    rho_lap_r_x = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,0]**2 * -1.))
    rho_lap_r_y = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,1]**2 * -1.))
    rho_lap_r_z = np.real(np.fft.ifftn(rho_g * g1_vector[:,:,:,2]**2 * -1.))

    rho_lap_r = rho_lap_r_x + rho_lap_r_y + rho_lap_r_z
    return(rho_lap_r)
