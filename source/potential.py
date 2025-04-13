import numpy as np
import scipy
from scipy import linalg
from scipy import special
from wavefunction_density import *


# # 计算hatree 势

# def cal_V_hatree(rho_r, g2_vector, grid_point):
    
#     rho_g = np.fft.fftn(rho_r) * 4 * np.pi
    
#     V_hatree_g = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex128)
    
#     V_hatree_g[g2_vector > 0 ] = rho_g[g2_vector > 0 ] / g2_vector[g2_vector > 0.]
    
#     V_hatree_r = np.real(np.fft.ifftn(V_hatree_g))
    
#     return(V_hatree_r)

# # 计算XC势

# def cal_V_tau(vtau, psi_g_3d, g1_vector):
#     "计算动能势"
#     V_tau = 0.+0.j
#     for psi_g in psi_g_3d:
#         dx_r = np.fft.ifftn(g1_vector[:,:,:,0] * psi_g * 1.j)
#         dy_r = np.fft.ifftn(g1_vector[:,:,:,1] * psi_g * 1.j) 
#         dz_r = np.fft.ifftn(g1_vector[:,:,:,2] * psi_g * 1.j)
        
#         dx_g = np.fft.fftn(dx_r * vtau) * g1_vector[:,:,:,0] * 1.j
#         dy_g = np.fft.fftn(dy_r * vtau) * g1_vector[:,:,:,1] * 1.j
#         dz_g = np.fft.fftn(dz_r * vtau) * g1_vector[:,:,:,2] * 1.j

#         V_tau = V_tau - 0.5*(dx_g + dy_g + dz_g)   
#     return(V_tau)



# def cal_V_xc(psi_g_3d, rho_r, g1_vector, vol, state_occupy_list, libxc_function_name = 'MGGA_X_R2SCAN'):
#     "计算xalpha势"
#     if libxc_function_name != '':
#         xc = pylibxc.LibXCFunctional(libxc_function_name, "unpolarized")
#         shape = rho_r.shape
#         inp = {}
#         inp["rho"] = np.ascontiguousarray(rho_r.reshape((-1,)))

#         if "GGA" in libxc_function_name :
#             rho_grad_r_x, rho_grad_r_y, rho_grad_r_z = cal_rhoe_grad(rho_r, g1_vector)
#             rho_grad_r = rho_grad_r_x**2 + rho_grad_r_y**2 + rho_grad_r_z**2
#             inp["sigma"] = np.ascontiguousarray(rho_grad_r.reshape((-1,)))
#             # inp["lapl"] = np.ascontiguousarray(rho_lap_r.reshape((-1,)))


#             if "MGGA" in libxc_function_name:
#                 kden = cal_knetic_density(psi_g_3d, g1_vector, vol, state_occupy_list)
#                 inp["tau"] = np.ascontiguousarray(kden.reshape((-1,)))

#             res = xc.compute(inp, do_exc=False, do_vxc=True)
#             V_xc = res['vrho'].reshape(shape)
#             vsigma = res['vsigma'].reshape(shape)

#             hx = vsigma * rho_grad_r_x
#             hy = vsigma * rho_grad_r_y
#             hz = vsigma * rho_grad_r_z
#             V_xc = V_xc  - 2.0*nabla_dot(hx,hy,hz, g1_vector)

#             if "MGGA" in libxc_function_name:
#                 vtau = res['vtau'].reshape(shape)
#                 V_tau = cal_V_tau(vtau, psi_g_3d, g1_vector)
#                 V_xc = V_xc + V_tau

#         else:
#             V_xc = res['vrho'].reshape(shape)
#     else:
#         alpha = 2. / 3.
#         V_xc = -1.* (3./2.* alpha) * np.cbrt(3. / np.pi * rho_r) / 2.

#     return(V_xc)

# # 计算动能势

# def cal_op_k(psi_g_3d, g2_vector):
    
#     V_kneltic = []
#     for i in psi_g_3d:
#         V_kneltic.append(g2_vector * i / 2.) 
    
#     V_kneltic = np.array(V_kneltic)
    
#     return(V_kneltic)


# def cal_V_ps_nloc(beta_nl, psi_g, ps_list, atom_symbol, grid_point):
    
#     V_Ps_nloc = []
#     for psi in psi_g:
#         tmp_v = np.zeros(grid_point, dtype = np.complex128)
#         for i in range(len(atom_symbol)):  # 每个原子
#             ps = ps_list[atom_symbol[i]]
#             for l in range(0, ps.l_max):  #每个层            
#                 for m in range(-l , l+1):            # 每个伸展方向                    
#                     for iprj in range(0, ps.n_proj[l] ): # 每个投影算符   
#                         for jprj in range(0, ps.n_proj[l]): # 每个投影算符
#                             tmp_v = tmp_v + ps.ps_proj[l][iprj, jprj] * beta_nl[i][l][m+l][iprj]* \
#                                             np.sum(np.conj(beta_nl[i][l][m+l][jprj]) * psi)
#         V_Ps_nloc.append(tmp_v)                   
                        
#     return(V_Ps_nloc)

# # 计算哈密顿量

# def cal_hamiltionian(V_loc_r ,psi_g_3d, rho_r, grid_point, g2_vector, g1_vector, vol, state_occupy_list):
    
#     V_xc_r = cal_V_xc(psi_g_3d, rho_r, g1_vector, vol, state_occupy_list)
#     V_hatree_r = cal_V_hatree(rho_r, g2_vector, grid_point)

    
#     V_loc_r = V_loc_r + V_xc_r + V_hatree_r 
    
#     return(V_loc_r)

# def op_V_loc(psi_g_3d, V_loc_r):
    
#     op_V_loc_g = []
#     for i in psi_g_3d:
    
#         psi_r =  np.fft.ifftn(i)

#         op_V_loc_r = V_loc_r * psi_r
    
#         tmp_V_loc_g = np.fft.fftn(op_V_loc_r)
    
#         op_V_loc_g.append(tmp_V_loc_g)
        
#     op_V_loc_g = np.array(op_V_loc_g)
    
#     return(op_V_loc_g)


# def op_H(V_loc_r, psi_g_3d, rho_r, beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list):
     
#     V_loc_r = cal_hamiltionian(V_loc_r ,psi_g_3d, rho_r, grid_point, g2_vector, g1_vector)
#     V_ps_nloc =  cal_V_ps_nloc(beta_nl, psi_g_3d, ps_list, atom_symbol, grid_point)
    
#     # 计算倒空间中的动能算符
#     op_k_g  = cal_op_k(psi_g_3d, g2_vector)     
    
#     # 计算倒空间中的局域算符
#     op_v_loc_g = op_V_loc(psi_g_3d, V_loc_r)
    
#     op_H_g = op_k_g + op_v_loc_g + V_ps_nloc
    
#     return(op_H_g)
    
    
    
    
