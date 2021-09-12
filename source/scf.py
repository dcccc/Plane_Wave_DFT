import numpy as np
import scipy
from scipy import linalg
from scipy import special

from wavefunction_density import *
from diag import *
from cal_E import *


# 自洽场

def scf(V_loc_r, psi_g, rho_r_all, beta_nl, g_vector, input):

    state_occupy_list = input.state_occupy_list
    grid_point    = input.grid_point
    atom_pos      = input.coord
    atom_num      = len(input.coord)
    atom_symbol   = input.atom_symbol
    vol           = input.vol
    atom_charge   = input.atom_charge
    atom_pos      = input.coord
    latt9         = input.latt9
    rec_latt      = input.rec_latt 
    ps_list       = input.ps_list
	
    gx_vector_mask = g_vector.gx_vector_mask
    g1_vector      = g_vector.g1_vector
    g2_vector      = g_vector.g2_vector
    n_gxw          = g_vector.n_gxw
    g_vector_mask  = g_vector.g_vector_mask




    
    n_state_all = len(state_occupy_list)
    n_point = np.prod(grid_point)
    
    eval0, psi_g_1d = diag_davidson(V_loc_r, psi_g, rho_r_all, beta_nl, g2_vector, grid_point,\
                                    gx_vector_mask, n_state_all, ps_list, atom_symbol)
   
    is_converged = False
    
    # 波函数归一化
    psi_g_1d  = orth1(psi_g_1d)
    psi_g_3d = []
    
    for i in psi_g_1d:
        psi_g_new = np.zeros(grid_point, dtype = np.complex)
        psi_g_new[gx_vector_mask] = i
        psi_g_3d.append(psi_g_new)
        
    psi_g_3d = np.array(psi_g_3d)
        
    rho_new = cal_rhoe(psi_g_3d,  vol, state_occupy_list, grid_point)
    
    rho_new = rho_r_all * 0.8 + rho_new * 0.2
    
    E_pw = cal_E_total(psi_g_3d, V_loc_r, beta_nl, input, g_vector)
    
    for ii in range(200):
    
        eval1, psi_g_1d = diag_davidson(V_loc_r, psi_g_3d, rho_new, beta_nl, g2_vector, grid_point,\
                                        gx_vector_mask, n_state_all, ps_list, atom_symbol)
    
        psi_g_1d  = orth1(psi_g_1d)
        psi_g_3d = []
        
        for i in psi_g_1d:
            psi_g_new = np.zeros(grid_point, dtype = np.complex)
            psi_g_new[gx_vector_mask] = i
            psi_g_3d.append(psi_g_new)            
        
        psi_g_3d = np.array(psi_g_3d)
        
        rho_new1 = cal_rhoe(psi_g_3d,  vol, state_occupy_list, grid_point)
               
        
        deval = np.sum(np.abs(eval1 - eval0))
        
        
        # 密度混合 

        rho_new = rho_new * 0.8 + rho_new1 * 0.2
        
        # rho_new = rho_new / np.sum(rho_new) * N_electron / vol * np.prod(grid_point) 

        d_rho  = np.sum(np.abs((rho_new - rho_new1)))
        
        eval0 = eval1
        
        
        E_pw_new = cal_E_total(psi_g_3d, V_loc_r, beta_nl, input, g_vector)
        

        E_total = np.sum(E_pw_new)
        dE_total = E_total - np.sum(E_pw)
        
        E_pw = E_pw_new
        
        print("scf:% 3d  dEtot=% 9.8f  Etot=% 9.8f   deval=% 9.8f     drho=% 7.5f" \
              %(ii+1, E_total, dE_total, deval, d_rho) )
        
        
        
        if dE_total < 0.0000001 and deval< 0.0000001 and d_rho < 0.0001:
            is_converged = True
            break    
        
        
    return(eval0, psi_g_3d, rho_new, E_pw_new , is_converged)
	
	
