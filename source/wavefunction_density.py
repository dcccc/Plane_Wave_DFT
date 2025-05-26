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


# guess rho


def atom_length(zion, znul):
  
    
    num_core_electron = znul - zion
    
    n_val = int(np.round(zion))
    
    length = 0.0 
    
    # For each set of core electron numbers, there are different decay lengths,
    # they start from nval=1, and proceed by group of 5, until a default is used 
    if n_val == 0.0:
        length = 0.0
    
    # Bare ions : adjusted on 1H and 2He only
    elif num_core_electron < 0.5 :
        data_length = [0.6, 0.4, 0.3, 0.25] + [0.0]*12
        length = 0.2
        if n_val <=4:
            length = data_length[n_val]
    
    # 1s2 core : adjusted on 3Li, 6C, 7N, and 8O
    elif num_core_electron < 2.5 :
        data_length = [1.8, 1.4, 1.0, 0.7, 0.6, 0.5, 0.4, 0.35] + [0.0]*8
        length = 0.3
        if n_val <= 8 :
            length = data_length[n_val]
    
    # Ne core (1s2 2s2 2p6) : adjusted on 11na, 13al, 14si and 17cl
    elif num_core_electron < 10.5 :
        data_length =  [2.0, 1.6, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7 , 0.7, 0.7] + [0.0]*6
        length = 0.6
        if n_val <= 10 :
            length = data_length[n_val]
    
    # Mg core (1s2 2s2 2p6 3s2) : adjusted on 19k, and on coreel==10
    elif num_core_electron < 12.5 :
        data_length =  [1.9, 1.5, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6 , 0.6, 0.6] + [0.0]*6
        length = 0.5
        if n_val <= 10 :
            length = data_length[n_val]
    
    # Ar core (Ne + 3s2 3p6) : adjusted on 20ca, 25mn and 30zn
    elif num_core_electron < 18.5 :
        data_length =  [2.0,  1.8,  1.5, 1.2,  1.0, 0.9,  0.85, 0.8, 0.75, 0.7,
                        0.65, 0.65] + [0.0]*4
        length = 0.6
        if n_val <= 12 :
            length = data_length[n_val]
            
    # Full 3rd shell core (Ar + 3d10) : adjusted on 31ga, 34se and 38sr
    elif num_core_electron < 28.5:
        data_length = [1.5 , 1.25, 1.15, 1.05, 1.00, 0.95, 0.95, 0.9 , 0.9 , 0.85,
                       0.85, 0.80, 0.8 , 0.75] + [0.0]*2
        length = 0.7
        if n_val <= 14:
            length = data_length[n_val]

    # Krypton core (Ar + 3d10 4s2 4p6) : adjusted on 39y, 42mo and 48cd
    elif num_core_electron < 36.5:
        data_length = [2.0 , 2.00, 1.60, 1.40, 1.25, 1.10, 1.00, 0.95, 0.90, 0.85,
                       0.80, 0.75] + [0.0]*4
        length = 0.7
        if n_val <= 12:
            length = data_length[n_val]
     
    

    # For the remaining elements, consider a function of nval only
    else:
        data_length= [2.0 , 2.00, 1.55, 1.25, 1.15, 1.10, 1.05, 1.0 , 0.95, 0.9,
                      0.85, 0.85] + [0.0]*4
        length = 0.8
        if nval <= 12:
            length = data_length[n_val]

    return(length)


element_symbol_list = ["H", "He", 
"Li", "Be", "B", "C", "N", "O", "F", "Ne", 
"Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
"K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
 "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", 
 "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
 "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
 "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", 
"Uup", "Lv", "Uus", "Uuo"]



def guess_rhoe(struct_factor, g2_vector, gx_vector_mask, atom_symbol, grid_point, ps_list, vol, N_electron):
   
    rho_g = np.zeros(grid_point, dtype=np.complex128)
    
    for sym in atom_symbol:
        element_idx = element_symbol_list.index(sym) + 1
        atom_length_v = atom_length(ps_list[sym].zval, element_idx)
        
        n_val = ps_list[sym].zval
        tmp_sf = struct_factor[sym]
        rho_g[gx_vector_mask] = rho_g[gx_vector_mask]  + n_val * tmp_sf[gx_vector_mask] * \
                                np.exp(-1.*g2_vector[gx_vector_mask]*atom_length_v**2)
            
    rho_r = np.real(np.fft.ifftn(rho_g))
    
    rho_r = rho_r / np.sum(rho_r) * N_electron / vol * np.prod(grid_point) 
                    
    return(rho_r)



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

def cal_lapl(rho_all, g2_vector): 
    
    rho_g = np.fft.fftn(rho_all)* g2_vector * -1.
    rho_lap_r = np.real(np.fft.ifftn(rho_g ))
    return(rho_lap_r)
