import numpy as np
import scipy
from scipy import linalg
from scipy import special

from wavefunction_density import *
from diag import *
from cal_E import *
import sys


# pulay mixing

def pulay_mix(rho_list, r_list, beta = 0.2 ): 
    
    n_mix = len(rho_list)
    
    r_array     = np.array(r_list)
    
    Lagrange_eq  = np.ones((n_mix+1, n_mix+1))
    Lagrange_eq[:-1,:-1]  = np.dot(r_array, r_array.T)
    Lagrange_eq[-1,-1]   = 0.
    
    z            = np.zeros(n_mix+1)
    z[-1]        = 1.
    
    result_fit=np.linalg.inv(Lagrange_eq).T
    alpha = result_fit[-1][:-1]

    alpha = alpha / np.sum(alpha)
    
    rho_new = 0.
    
    for i in range(n_mix):
        rho_new = rho_new +   (rho_list[i] + beta * r_list[i])* alpha[i]

    return(rho_new )

# 自洽场

def scf(V_loc_r, psi_g, rho_r_all, beta_nl, g_vector, input, libxc):

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
    

    is_need_tau = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_TAU  for xc in libxc]) > 0 
    is_need_lap = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_LAPLACIAN  for xc in libxc]) > 0 
    is_gga = np.sum([xc.get_family() & pylibxc.flags.XC_FAMILY_GGA  for xc in libxc]) > 0 


    rho_grad_r = 0.0
    kden       = 0.0
    rho_lap_r  = 0.0

    if is_gga or is_need_lap or is_need_tau:
        rho_grad_r = cal_rhoe_grad(rho_r_all, g1_vector)
    if is_need_lap:
        rho_lap_r = cal_rhoe_lap(rho_r_all, g2_vector)
    if is_need_tau:
        kden = cal_knetic_density(psi_g, g1_vector, vol, state_occupy_list)

    rho_value = [rho_r_all, rho_grad_r, kden, rho_lap_r]

    eval0, psi_g_1d = diag_davidson(V_loc_r, psi_g, rho_value, beta_nl, g2_vector, g1_vector, grid_point,\
                                    gx_vector_mask, n_state_all, ps_list, atom_symbol, vol, state_occupy_list, libxc)
   
    is_converged = False
    
    # 波函数归一化
    psi_g_1d  = orth1(psi_g_1d)
    psi_g_3d = []
    
    for i in psi_g_1d:
        psi_g_new = np.zeros(grid_point, dtype = np.complex128)
        psi_g_new[gx_vector_mask] = i
        psi_g_3d.append(psi_g_new)
        
    psi_g_3d = np.array(psi_g_3d)
        
    rho_new = cal_rhoe(psi_g_3d,  vol, state_occupy_list, grid_point)
    
    rho_new = rho_r_all * 0.8 + rho_new * 0.2

    
    E_pw = cal_E_total(psi_g_3d, V_loc_r, rho_value, beta_nl, input, g_vector, libxc=libxc)
    
    rho_list = [rho_new.reshape((-1,)).copy() ]
    r_list   = [rho_new.reshape((-1,)) -rho_r_all.reshape((-1,)) ]
    f_list   = []

    if is_need_tau:
        kden_new = cal_knetic_density(psi_g_3d, g1_vector, vol, state_occupy_list)*0.2 + kden * 0.8
        kden_list = [kden_new]
        kden_r_list = [kden_new.reshape((-1,)) - kden.reshape((-1,)) ]



    N_electron = np.sum(atom_charge)

    for ii in range(200):
        
        psi_g_1d = np.array([i[gx_vector_mask] for i in psi_g_3d])
        psi_g_1d  = orth1(psi_g_1d)
        psi_g_3d = []
        for i in psi_g_1d:
            psi_g_new = np.zeros(grid_point, dtype = np.complex128)
            psi_g_new[gx_vector_mask] = i
            psi_g_3d.append(psi_g_new)            
        
        psi_g_3d = np.array(psi_g_3d)


        if is_gga:
            rho_grad_r = cal_rhoe_grad(rho_new, g1_vector)
        if is_need_lap:
            rho_lap_r = cal_rhoe_lap(rho_new, g2_vector)
        if is_need_tau:
            kden = cal_knetic_density(psi_g_3d, g1_vector, vol, state_occupy_list)

        rho_value = [rho_new, rho_grad_r, kden, rho_lap_r]


        eval1, psi_g_1d = diag_davidson(V_loc_r, psi_g_3d, rho_value, beta_nl, g2_vector, g1_vector, grid_point,\
                                        gx_vector_mask, n_state_all, ps_list, atom_symbol, vol, state_occupy_list, libxc)

        psi_g_1d = orth1(psi_g_1d)
        psi_g_3d = []
        
        for n,i in enumerate(psi_g_1d):
            psi_g_new = np.zeros(grid_point, dtype = np.complex128)
            psi_g_new[gx_vector_mask] = i
            psi_g_3d.append(psi_g_new)            
        
        psi_g_3d = np.array(psi_g_3d)
        
        rho_new1 = cal_rhoe(psi_g_3d,  vol, state_occupy_list, grid_point)
        
        deval = np.sum(np.abs(eval1 - eval0))
        
        
        # 密度混合        
        rho_list = rho_list + [rho_new1.reshape((-1,)).copy() ]
        r_list   = r_list + [rho_new1.reshape((-1,)).copy() - rho_new.reshape((-1,)).copy() ]
        f_list   = f_list + [r_list[-1] - r_list[-2] ]

        # if ii == 10:
        #     print(" start pulay mixing ")
        # if ii >= 10 :
        #     rho_new = pulay_mix(rho_list, r_list, beta = 0.2 )
        #     rho_new = rho_new.reshape(grid_point)
        #     rho_new = rho_new / np.sum(rho_new) * N_electron / vol * np.prod(grid_point) 
            
        # elif ii< 10 :
        rho_new = rho_new * 0.8 + rho_new1 * 0.2



        # rho_new[rho_new < 1.e-8] = 0.0

        rho_new = rho_new / np.sum(rho_new) * np.sum(state_occupy_list)*2 / vol * np.prod(grid_point) 

        d_rho  = np.sum(np.abs((rho_new - rho_new1)))
        
        eval0 = eval1
        
        rho_value = [rho_new, rho_grad_r, kden, rho_lap_r]
        E_pw_new = cal_E_total(psi_g_3d, V_loc_r, rho_value, beta_nl, input, g_vector, libxc)
        

        E_total = np.sum(E_pw_new)
        dE_total = E_total - np.sum(E_pw)
        
        E_pw = E_pw_new
        
        print("scf:% 3d  Etot=% 9.8f  dEtot=% 9.8f   deval=% 9.8f     drho=% 7.5f" \
              %(ii+1, E_total, dE_total, deval, d_rho) )
        
        
        
        if dE_total < 0.000001 and deval< 0.000001 and d_rho < 0.0001:
            is_converged = True
            break    
        
    return(eval0, psi_g_3d, rho_new, E_pw_new , is_converged)
	
	
