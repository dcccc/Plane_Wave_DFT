import numpy as np
import scipy
from scipy import linalg
from scipy import special

from wavefunction_density import *
from cal_E import *
from potential import *


# 计算能量

#计算动能

def cal_kinetic_e(psi_g, g2_vector, state_occupy_list ):
    
    E_kinectic = 0.0
    for i,occupy  in enumerate(state_occupy_list):
    
        E_kinectic = E_kinectic + np.sum(np.conj(psi_g[i]) * g2_vector * psi_g[i] ).real * occupy
    
    return(E_kinectic)

# xalpha XC能

def cal_E_xc(rho_r, vol , n_point):
    
    alpha = 2. / 3.

    E_xc = -9./8.* alpha * np.cbrt(3./ np.pi) * np.sum( rho_r**(4./3.))
    
    E_xc = E_xc * vol / n_point * 0.5
    
    return(E_xc)

# 计算Hatree势能

def cal_E_hatree(rho_r, g2_vector, g_vector_mask, vol, n_point, grid_point):
 
    rho_g = np.conj(np.fft.fftn(rho_r)  / n_point) 
   
    V_hatree_r = cal_V_hatree(rho_r.copy(), g2_vector.copy(), grid_point )
    V_hatree_g = np.fft.fftn(V_hatree_r)
    
    E_hatree =  np.sum(rho_g[g_vector_mask][1:] * V_hatree_g[g_vector_mask][1:] ).real
   
    E_hatree = E_hatree * vol / n_point / 2.
    
    return(E_hatree)

# 局域能

def cal_E_loc(V_loc_r, rho_r, g2_vector, g_vector_mask, vol, grid_point ):
    
    n_point = np.prod(grid_point)
    
    V_loc_g = np.fft.fftn(V_loc_r)
    rho_g = np.conj(np.fft.fftn(rho_r)  / n_point) 
    
    E_loc_g = np.sum(V_loc_g[g_vector_mask][1:] * rho_g[g_vector_mask][1:]).real
    
    E_loc_g =  E_loc_g * vol / n_point
    
    return(E_loc_g)

# ps core能量

def cal_pspcore_E(atom_symbol, atom_charge, ps_list, vol):
    
    
    atom_core = {}
    
    for ps in ps_list.values():
        c1,c2,c3,c4 = ps.c_loc
        r_loc = ps.r_loc
        zval  = ps.zval
        
        tmp_v = 2. * np.pi * zval * r_loc**2 + (2.* np.pi)**1.5 * r_loc**3 * \
                                               (c1 + 3.* c2 + 15. * c3 + 105. * c4)
        atom_core[ps.atom_specie] = tmp_v
    
    
    tot_charge = np.sum(atom_charge)
    E_pspcore  = 0.0
    
    for n, i in enumerate(atom_symbol):
        E_pspcore = E_pspcore + atom_core[i]
    
    E_pspcore = tot_charge * E_pspcore / vol
    
    return(E_pspcore)

# ps非局域能

def cal_E_ps_nloc(beta_nl, psi_g, ps_list, atom_symbol, state_occupy_list):
    
    e_Ps_nloc = 0.
    for n, occupy in enumerate(state_occupy_list):
        psi = psi_g[n]
        for i in range(len(atom_symbol)):  # 每个原子
            ps = ps_list[atom_symbol[i]]
            for l in range(0, ps.l_max):  #每个层            
                for m in range(-l , l+1):            # 每个伸展方向
                    tmp_beta_psi = np.zeros(ps.n_proj[l], dtype = np.complex)
                    for iprj in range(0, ps.n_proj[l] ):
                        tmp_beta_psi[iprj] = np.sum(np.conj(beta_nl[i][l][m+l][iprj]) * psi)
                    
                    
                    for iprj in range(0, ps.n_proj[l] ): # 每个投影算符   
                        for jprj in range(0, ps.n_proj[l]): # 每个投影算符
                            e_Ps_nloc = e_Ps_nloc + ps.ps_proj[l][iprj, jprj] * \
                                                    np.real(np.conj(tmp_beta_psi[iprj]) * tmp_beta_psi[jprj]) *\
                                                    occupy
    e_Ps_nloc = e_Ps_nloc * 2.  
                        
    return(e_Ps_nloc)


# ewald 能


def cal_ewald_sum(atom_pos, latt9, rec_latt, atom_charge):
    
    vol = np.linalg.det(latt9)
    
    rec_lat = np.linalg.inv(latt9).T * 2. * np.pi
    
    len_latt    = np.linalg.norm(latt9, axis = 1)
    len_reclatt = np.linalg.norm(rec_lat, axis = 1)
    
    atom_num = len(atom_pos)
    
    g_cut = 2.
    e_cut = 1.e-8
    
    yita  = (g_cut**2 / np.log(e_cut) * -1.)**0.5 / 2.
    

    
    # 自相互作用能
    e_self = -2. * yita * np.sum(atom_charge**2) / np.pi**0.5  \
             - np.pi * np.sum(atom_charge)**2 / (vol * yita**2)
    

    # 实空间部分
    
    e_real = 0.0
    
    tmax = (0.5 * np.log(e_cut) * -1.)**0.5 / yita
    
    n1 = int(np.around(tmax / len_latt[0] + 1.5 ))
    n2 = int(np.around(tmax / len_latt[1] + 1.5 ))
    n3 = int(np.around(tmax / len_latt[2] + 1.5 ))
    
    
    
    for i in range(atom_num):
        for j in range(atom_num):
            qij = atom_charge[i] * atom_charge[j]
            
            posij = atom_pos[i] - atom_pos[j]
            for ii in range(-n1,n1+1):
                for jj in range(-n2,n2+1):
                    for kk in range(-n3,n3+1):
                        if i != j or abs(ii) + abs(jj) + abs(kk) != 0:
                            t1 = ii*latt9[0] + jj*latt9[1] + kk*latt9[2]
                            rmag2 = np.linalg.norm(t1 - posij)
                            e_real =  e_real + qij * special.erfc(rmag2 * yita ) / rmag2
                            
                            
    # 倒空间部分
    
    e_rec = 0.0 
    
    m1 = int(np.around(g_cut / len_reclatt[0] + 1.5 ))
    m2 = int(np.around(g_cut / len_reclatt[1] + 1.5 ))
    m3 = int(np.around(g_cut / len_reclatt[2] + 1.5 ))
    
    
    for ii in range(-m1,m1+1):
        for jj in range(-m2,m2+1):
            for kk in range(-m3,m3+1):
                if abs(ii) + abs(jj) + abs(kk) != 0:
                    g  = ii * rec_latt[0]  + jj * rec_latt[1] + kk * rec_latt[2]                
                    g2 =  np.sum(g * g)
                    x  = 4. * np.pi / vol * np.exp(-0.25 * g2 / yita**2 ) / g2
                    
                    for i in range(atom_num):
                        for j in range(atom_num):
                            qij = atom_charge[i] * atom_charge[j]
                            posij = atom_pos[i] - atom_pos[j]
                            gtau  = np.sum(g * posij)
                            
                            e_rec = e_rec + x * qij * np.cos(gtau)
                            
    e_wald = (e_self + e_real + e_rec ) * 0.5 
    
    
    return(e_wald)

# 总能

def cal_E_total(psi_g, V_loc_r, beta_nl, input, g_vector):
    
	
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
	
	
	
    n_point    = np.prod(grid_point)
    
    rho_r      = cal_rhoe(psi_g,  vol, state_occupy_list, grid_point)

    E_kinectic = cal_kinetic_e(psi_g, g2_vector, state_occupy_list )

    E_xc       = cal_E_xc(rho_r, vol , n_point)
    
    E_hatree   = cal_E_hatree(rho_r, g2_vector, g_vector_mask.copy(), vol, n_point, grid_point)    
    
    E_loc      = cal_E_loc(V_loc_r, rho_r, g2_vector, g_vector_mask, vol, n_point )
    
    E_ps_nl    = cal_E_ps_nloc(beta_nl, psi_g, ps_list, atom_symbol, state_occupy_list) 
    
    E_pscore   = cal_pspcore_E(atom_symbol, atom_charge, ps_list, vol)
    
    E_NN       = cal_ewald_sum(atom_pos, latt9, rec_latt, atom_charge) 
    
    E_total    = [E_kinectic, E_xc, E_hatree, E_loc, E_ps_nl, E_pscore, E_NN] 
    
    
    
    
    return(E_total)
