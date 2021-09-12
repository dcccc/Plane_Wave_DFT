import numpy as np
import scipy
from scipy import linalg
from scipy import special




# 计算hatree 势

def cal_V_hatree(rho_r, g2_vector, grid_point):
    
    rho_g = np.fft.fftn(rho_r) * 4 * np.pi
    
    V_hatree_g = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex)
    
    V_hatree_g[g2_vector > 0 ] = rho_g[g2_vector > 0 ] / g2_vector[g2_vector > 0.]
    
    V_hatree_r = np.real(np.fft.ifftn(V_hatree_g))
    
    return(V_hatree_r)

# 计算XC势

def cal_V_xc(rho_r):
    "计算xalpha势"

    alpha = 2. / 3.

    V_xc = -1.* (3./2.* alpha) * np.cbrt(3. / np.pi * rho_r) / 2.
    
    return(V_xc)

# 计算动能势

def cal_op_k(psi_g_3d, g2_vector):
    
    V_kneltic = []
    for i in psi_g_3d:
        V_kneltic.append(g2_vector * i / 2.) 
    
    V_kneltic = np.array(V_kneltic)
    
    return(V_kneltic)


def cal_V_ps_nloc(beta_nl, psi_g, ps_list, atom_symbol):
    
    V_Ps_nloc = []
    for psi in psi_g:
        tmp_v = np.zeros(grid_point, dtype = np.complex)
        for i in range(len(atom_symbol)):  # 每个原子
            ps = ps_list[atom_symbol[i]]
            for l in range(0, ps.l_max):  #每个层            
                for m in range(-l , l+1):            # 每个伸展方向                    
                    for iprj in range(0, ps.n_proj[l] ): # 每个投影算符   
                        for jprj in range(0, ps.n_proj[l]): # 每个投影算符
                            tmp_v = tmp_v + ps.ps_proj[l][iprj, jprj] * beta_nl[i][l][m+l][iprj]* \
                                            np.sum(np.conj(beta_nl[i][l][m+l][jprj]) * psi)
        V_Ps_nloc.append(tmp_v)                   
                        
    return(V_Ps_nloc)

# 计算哈密顿量

def cal_hamiltionian(V_loc_r , rho_r, grid_point, g2_vector):
    
    V_xc_r = cal_V_xc(rho_r)
    V_hatree_r = cal_V_hatree(rho_r, g2_vector, grid_point)

    
    V_loc_r = V_loc_r + V_xc_r + V_hatree_r 
    
    return(V_loc_r)

def op_V_loc(psi_g_3d, V_loc_r):
    
    op_V_loc_g = []
    for i in psi_g_3d:
    
        psi_r =  np.fft.ifftn(i)

        op_V_loc_r = V_loc_r * psi_r
    
        tmp_V_loc_g = np.fft.fftn(op_V_loc_r)
    
        op_V_loc_g.append(tmp_V_loc_g)
        
    op_V_loc_g = np.array(op_V_loc_g)
    
    return(op_V_loc_g)


def op_H(V_loc_r, psi_g_3d, rho_r, beta_nl, g2_vector, grid_point, ps_list, atom_symbol):
     
    V_loc_r = cal_hamiltionian(V_loc_r , rho_r, grid_point, g2_vector)
    V_ps_nloc = V_ps_nloc  = cal_V_ps_nloc(beta_nl, psi_g_3d, ps_list, atom_symbol)
    
    # 计算倒空间中的动能算符
    op_k_g  = cal_op_k(psi_g_3d, g2_vector)     
    
    # 计算倒空间中的局域算符
    op_v_loc_g = op_V_loc(psi_g_3d, V_loc_r)
    
    op_H_g = op_k_g + op_v_loc_g + V_ps_nloc
    
    return(op_H_g)
    
    
    
    
