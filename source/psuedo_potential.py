import numpy as np
import scipy
from scipy import linalg
from scipy import special


# psuedo potential local potential

def eval_v_psloc_g(g2, c1, c2, c3, c4, r_loc, pre1, pre2):
    
    gr = g2**0.5 * r_loc
    expgr2 = np.exp(-0.5 * gr**2)
    
    tmp_g = pre1 / g2*expgr2 + pre2 * expgr2 * (c1 + 
                                                c2*(3.  - gr**2)+
                                                c3*(15. - 10.0*gr**2 + gr**4)+
                                                c4*(105.- 105.*gr**2 + 21.*gr**4 - gr**6))
    
    return(tmp_g)


def cal_ps_vloc_r(ps, g_vector, input, struct_factor):
    
    grid_point = input.grid_point
    vol        = input.vol
    
    g_vector_mask = g_vector.g_vector_mask
    g2_vector     = g_vector.g2_vector
    
    n_point = np.prod(grid_point)
    
    ps_v_loc_g = np.zeros(grid_point, dtype=np.complex)
    
    pre1 = -4. * np.pi * ps.zval
    pre2 = (8.0 * np.pi**3)**0.5 * ps.r_loc**3
    
    c1,c2,c3,c4 = ps.c_loc
    r_loc = ps.r_loc
    
    
    
    for i in range(grid_point[0]):
        for j in range(grid_point[1]):
            for k in range(grid_point[2]):
                if g_vector_mask[i, j, k]:
                    g2 = g2_vector[i,j,k]
                    if g2 > 0.0:                    
                        tmp_g = eval_v_psloc_g(g2, c1, c2, c3, c4, r_loc, pre1, pre2 )
                    else:
                        tmp_g = 0.0
                    ps_v_loc_g[i,j,k] = tmp_g
                    
    ps_v_loc_g = ps_v_loc_g * struct_factor / vol
    ps_v_loc_c = np.fft.ifftn(ps_v_loc_g) * n_point

    return(ps_v_loc_c)



def get_ps_vloc_r(ps_list, g_vector,input, struct_factor ): 



    
    atom_symbol = input.atom_symbol   
    
    v_loc_r = 0.
    for sym in set(atom_symbol):
        v_loc_r = v_loc_r + cal_ps_vloc_r(ps_list[sym], g_vector, input, struct_factor[sym])
        
    
    return(v_loc_r)

# psuedo potential nonlocal potential

def ylm_real(l, m ,r_list):
    
    r1, r2, r3 = r_list
    
    rmod = (r1**2 + r2**2 + r3**2)**0.5

    if rmod < 1.0e-7:
        cost = 0.0
    else:
        cost = r3/rmod
        
    if r1 > 1.0e-7 :
        phi = np.arctan(r2 / r1)
    elif r1 < -1.0e-7:
        phi = np.arctan(r2 / r1) + np.pi
    else:
        phi = np.pi * np.sign(r2) / 2.0
    # print(phi)
    sint = (max([0.0, 1 - cost**2] ))**0.5
    
    ylm = 0.0
    
    if l == 0:
        ylm = 0.5 * (1.0 / np.pi)**0.5
    elif l == 1:
        if m == -1:
            ylm = 0.5 * (3.0 / np.pi )**0.5 * sint * np.sin(phi)            
        elif m == 0:
            ylm = 0.5 * (3.0 / np.pi )**0.5 * cost
        elif m == 1:
            ylm = 0.5 * (3.0 / np.pi )**0.5 * sint * np.cos(phi)
            
    elif l == 2:
        if m == -2 :
            ylm = (15./16./np.pi)**0.5 * sint**2 * np.sin(2. * phi)
        elif m== -1 :
            ylm = (15./4./np.pi )**0.5 * cost*sint * np.sin(phi)
        elif m == 0:
            ylm = 0.25 * (5./np.pi )**0.5 * (3. * cost**2 - 1.)
        elif m == 1:
            ylm = (15./4./np.pi )**0.5 * cost*sint * np.cos(phi)
        elif m == 2:
            ylm = 0.5 * (15./4./np.pi )**0.5 * sint**2 * np.cos(2. * phi)
        
    elif l == 3:
        if m == -3 :
            ylm = 0.25*(35./2./np.pi)**0.5 * sint**3 * np.sin(3.0*phi)
        elif m == -2 :
            ylm = 0.25*(105./np.pi)**0.5 * sint**2 * cost * np.sin(2.0*phi)
        elif m == -1:
            ylm = 0.25*(21./2./np.pi)**0.5 * sint* (5.*cost**2 - 1.)  * np.sin(phi)
        elif m == 0:
            ylm = 0.25*(7./np.pi)**0.5 * (5.*cost**3 - 3.*cost)
        elif m == 1 :
            ylm = 0.25*(21./2./np.pi)**0.5 * sint* (5.*cost**2 - 1.)  * np.cos(phi)
        elif m == 2:
            ylm = 0.25*(105./np.pi)**0.5 * sint**2 * cost * np.cos(2.0*phi)
        elif m == 3:
            ylm = 0.25*(35./2./np.pi)**0.5 * sint**3 * np.cos(3.0*phi)
            
    return(ylm)

def eval_proj_g(ps, l, iprj, gm, vol):
    
    vprj = 0.0
    rrl = ps.r_cut[l]
    
    gr2 = (gm * rrl)**2
    
   
    if l == 0:
        if iprj == 1:            
            vprj = np.exp(-0.5 * gr2)
        elif iprj == 2 :
            vprj = 2. / (15.)**0.5 * np.exp(-0.5 * gr2) * (3.0 - gr2)
        elif iprj == 3 :
            vprj = (4. / 3.) / (105.)**0.5 * np.exp(-0.5 * gr2) * (15.0 - 10. * gr2 + gr2**2)
            
    if l == 1:
        if iprj == 1: 
            vprj = (1. / (3.)**0.5) * np.exp(-0.5 * gr2) * gm            
        elif iprj == 2: 
            vprj = (2. / (105.)**0.5) * np.exp(-0.5 * gr2) * gm * (5. - gr2)            
        elif iprj == 3: 
            vprj = (4. / 3.) / (1155.)**0.5 * np.exp(-0.5 * gr2)  * gm * (35.0 - 14. * gr2 + gr2**2)            
            
    if l == 2 :
        if iprj ==1:
            vprj = (1. / 15.**0.5) *  np.exp(-0.5 * gr2) * gm**2
        elif iprj ==2:
            vprj = (2. / 3.)/ (105.)**0.5 * np.exp(-0.5 * gr2) * gm**2 * (7. - gr2)
            
    elif l == 3:
        vprj = gm**3 * np.exp(-0.5 * gr2)* 16. / 105.**0.5 / 8. / 2.
        
    
    pre = 4. * np.pi**(5. / 4.) * (2.**(l + 1) * rrl**(2 * l + 3) / vol)**0.5
    
    vprj = pre * vprj
    
    return(vprj)
        

def check_beta_nl_norm(beta_nl, input, ps_list):
    
    
    print("atom num, l, iprj, m   int_real_prj   int_imag_prj   int_prj   norm_g_real   norm_g_imag")
    
    for i in range(2):  # 每个原子
        tmp_atom = []
        for l in range(0, ps.l_max):  #每个层
            tmp_l =[]
            for m in range(-l , l+1):            # 每个伸展方向
                tmp_iprj = []
                for iprj in range(1, ps.n_proj[l] + 1): # 每个投影算符    
                    tmp_beta_nl_g = beta_nl[i][l][l+m][iprj - 1]
                    norm_g = np.sum(np.conj(tmp_beta_nl_g) * tmp_beta_nl_g)
                    tmp_beta_nl_r = np.fft.ifftn(tmp_beta_nl_g) * n_point
                    int_real_prj = np.sum(np.real(tmp_beta_nl_r)**2) * vol / n_point
                    int_imag_prj = np.sum(np.imag(tmp_beta_nl_r)**2) * vol / n_point
                    int_prj      = np.sum(np.conj(tmp_beta_nl_r) * tmp_beta_nl_r) *  vol / n_point
                    print("%d, %d, %d, %d   % 10.8f   % 10.8f   % 10.8f   % 10.8f   % 10.8f" \
                          %(i,l,m,iprj, int_real_prj, int_imag_prj, int_prj, np.real(norm_g), np.imag(norm_g)))
                    
 
def get_ps_vnloc_g(input, g_vector, ps_list):  
    
    grid_point    = input.grid_point
    atom_pos      = input.coord
    atom_num      = len(input.coord)
    atom_symbol   = input.atom_symbol
    vol           = input.vol
    
    
    gx_vector_mask = g_vector.gx_vector_mask
    g1_vector      = g_vector.g1_vector
    
    n_point = np.prod(grid_point)
    beta_nl = []
    
    
    for i in range(atom_num):  # 每个原子
        tmp_atom = []
        ps =ps_list[atom_symbol[i]] 
        for l in range(0, ps.l_max):  #每个层
            tmp_l =[]
            for m in range(-l , l+1):            # 每个伸展方向
                tmp_iprj = []
                for iprj in range(1, ps.n_proj[l] + 1): # 每个投影算符     
                    tmp_m = np.zeros(grid_point, dtype = np.complex)
    
                    for ii in range(grid_point[0]):
                        for jj in range(grid_point[1]):
                            for kk in range(grid_point[2]):                            
                                if gx_vector_mask[ii,jj,kk] :
                                    g = g1_vector[ii,jj,kk]
                                    gm = np.linalg.norm(g)
                                    gx = np.sum(atom_pos[i] * g)
                                    sf = np.cos(gx) - np.sin(gx) * 1.j
                                    tmp_m[ii,jj,kk] = ylm_real(l, m, g) * eval_proj_g(ps, l, iprj, gm, vol) * sf
                    tmp_iprj.append(tmp_m)            
                tmp_l.append(tmp_iprj)
            tmp_atom.append(tmp_l)
        beta_nl.append(tmp_atom)
    
    return(beta_nl)
