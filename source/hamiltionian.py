import numpy as np
import scipy
from scipy import linalg
from scipy import special
from wavefunction_density import *

try:
    import pylibxc
    libxc_installed = True
except:
    print("pylibxc is not installed!")
    libxc_installed = False


# 计算hatree 势

def cal_V_hatree(rho_r, g2_vector, grid_point):
    
    rho_g = np.fft.fftn(rho_r) * 4 * np.pi
    
    V_hatree_g = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex128)
    
    V_hatree_g[g2_vector > 0 ] = rho_g[g2_vector > 0 ] / g2_vector[g2_vector > 0.]
    
    V_hatree_r = np.real(np.fft.ifftn(V_hatree_g))
    
    return(V_hatree_r)

# 计算XC势
def cal_V_tau(vtau, psi_g_3d, g1_vector):
    "计算动能势"
    V_tau = np.zeros(psi_g_3d.shape, dtype = np.complex128)
    if not isinstance(vtau, str):
        for n, psi_g in enumerate(psi_g_3d):
            dx_r = np.fft.ifftn(g1_vector[:,:,:,0] * psi_g * 1.j)
            dy_r = np.fft.ifftn(g1_vector[:,:,:,1] * psi_g * 1.j)
            dz_r = np.fft.ifftn(g1_vector[:,:,:,2] * psi_g * 1.j)
        
            dx_g = np.fft.fftn(dx_r * vtau) * g1_vector[:,:,:,0] * 1.j
            dy_g = np.fft.fftn(dy_r * vtau) * g1_vector[:,:,:,1] * 1.j
            dz_g = np.fft.fftn(dz_r * vtau) * g1_vector[:,:,:,2] * 1.j
            
            V_tau[n] = - 0.5*(dx_g + dy_g + dz_g)

    
    return(V_tau)



def cal_V_xc(rho_value, g1_vector, g2_vector, vol, state_occupy_list, libxc = []):
    "计算xalpha势"
    rho_r, rho_grad_r, kden, rho_lapl = rho_value

    shape = rho_r.shape
    V_xc  = 0.0
    vtau  = 0.0
    if libxc != [] and libxc_installed:      
        inp = {}
        inp["rho"] = np.ascontiguousarray(rho_r.reshape((-1,)))

        is_need_tau = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_TAU  for xc in libxc]) 
        is_need_lap = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_LAPLACIAN  for xc in libxc]) 
        is_gga = np.sum([xc.get_family() & pylibxc.flags.XC_FAMILY_GGA  for xc in libxc])

        if is_gga or is_need_lap or is_need_tau:
            rho_grad_r_x , rho_grad_r_y , rho_grad_r_z = rho_grad_r 
            rho_grad_r_total = rho_grad_r_x**2 + rho_grad_r_y**2 + rho_grad_r_z**2
            inp["sigma"] = np.ascontiguousarray(rho_grad_r_total.reshape((-1,)))

        if is_need_tau:
            inp["tau"] = np.ascontiguousarray(kden.reshape((-1,)))
        else:
            vtau = "none"
        if is_need_lap:
            inp["lapl"] = np.ascontiguousarray(rho_lapl.reshape((-1,)))

        for xc in libxc:   
            res = xc.compute(inp, do_exc=False, do_vxc=True)
            V_xc += res['vrho'].reshape(shape)

            if is_gga or is_need_lap or is_need_tau:
                vsigma = res['vsigma'].reshape(shape)
                hx = vsigma * rho_grad_r_x
                hy = vsigma * rho_grad_r_y
                hz = vsigma * rho_grad_r_z
                V_xc = V_xc  - 2.0*nabla_dot(hx,hy,hz, g1_vector)


            if is_need_tau:
                vtau += res['vtau'].reshape(shape)
            if is_need_lap :
                vlapl = res['vlapl'].reshape(shape)
                vlapl[np.isnan(vlapl)] = 0.0
                V_xc = V_xc - np.real(np.fft.ifftn(np.fft.fftn(vlapl)*g2_vector))

    else:
        alpha = 2. / 3.
        V_xc = -1.* (3./2.* alpha) * np.cbrt(3. / np.pi * rho_r) / 2.

    return(V_xc, vtau)

# 计算动能势

def cal_op_k(psi_g_3d, g2_vector):
    
    V_kneltic = []
    for i in psi_g_3d:
        V_kneltic.append(g2_vector * i / 2.) 
    
    V_kneltic = np.array(V_kneltic)
    
    return(V_kneltic)


def cal_V_ps_nloc(beta_nl, psi_g, ps_list, atom_symbol, grid_point):
    
    V_Ps_nloc = []
    for psi in psi_g:
        tmp_v = np.zeros(grid_point, dtype = np.complex128)
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

def cal_hamiltionian(V_loc_r , psi_g_3d,  rho_r, grid_point, g2_vector, g1_vector, vol, state_occupy_list):
    
    V_hatree_r = cal_V_hatree(rho_r, g2_vector, grid_point)


    V_loc_r = V_loc_r + V_hatree_r 
    
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


def op_H(V_loc_r, V_xc_all, psi_g_3d, rho_r, beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list):
     
    V_loc_r0 = cal_hamiltionian(V_loc_r, psi_g_3d, rho_r, grid_point, g2_vector, g1_vector, vol, state_occupy_list)
    V_ps_nloc  = cal_V_ps_nloc(beta_nl, psi_g_3d, ps_list, atom_symbol, grid_point)
    

    V_xc_r, vtau_r =  V_xc_all

    # 计算metagga中的动能算符
    op_tau = cal_V_tau(vtau_r, psi_g_3d, g1_vector)

    # 计算倒空间中的动能算符
    op_k_g  = cal_op_k(psi_g_3d, g2_vector)     
    
    # 计算倒空间中的局域算符
    op_v_loc_g = op_V_loc(psi_g_3d, V_loc_r0 + V_xc_r)
    
    op_H_g = op_k_g + op_v_loc_g + V_ps_nloc + op_tau

    return(op_H_g)
    
    
    
    
