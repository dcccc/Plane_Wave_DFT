import numpy as np
import scipy,copy
from scipy import linalg
from scipy import special


from hamiltionian import *

# 对角化

def Hermitian(h):
    return(np.triu(h) + np.conj(np.triu(h)- np.diag(np.diag(h))).T  - 1.j*np.diag(np.diag(h)).imag)


def diag_davidson(V_loc_r, psi_g_3d, rho_value, beta_nl, g2_vector, g1_vector, \
                  grid_point, gx_vector_mask, n_state_all, ps_list, atom_symbol, vol, state_occupy_list, libxc):

    V_xc_all = cal_V_xc(rho_value, g1_vector, g2_vector, vol, state_occupy_list, libxc = libxc)

    HX = op_H(V_loc_r, V_xc_all, psi_g_3d, rho_value[0], beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
    HX_g_1d  = np.array([i[gx_vector_mask] for i in HX])
    psi_g_1d = np.array([i[gx_vector_mask] for i in psi_g_3d])

    eval0 = np.array([np.real(np.sum(np.conj(i) * j)) for i, j in zip(psi_g_1d, HX_g_1d) ])

    R   = np.array([eval0[i] * psi_g_1d[i] - HX_g_1d[i]  for i in range(len(eval0))])

    res = np.array([np.real(np.sum(np.dot(np.conj(i), i))**0.5)  for i in R])
    
    for ii in range(100):
        
        res_norm = 1.0 / res
        R = np.array([res_norm[i] * R[i] for i in range(n_state_all)])
        R = np.array([i / ( g2_vector[gx_vector_mask] + 12.) for i in R])

        R_3d  = []
        HR_1d = []
        for i in range(n_state_all):
            tmp_3d = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex128)
            tmp_3d[gx_vector_mask] = R[i]
            R_3d.append(tmp_3d)
        
        R_3d  = np.array(R_3d)       
        V_xc_all = cal_V_xc(rho_value, g1_vector, g2_vector, vol, state_occupy_list, libxc = libxc)
        HR    = op_H(V_loc_r,V_xc_all, R_3d, rho_value[0], beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
        HR_1d = np.array([i[gx_vector_mask]  for i in HR])
        
        
        Hred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex128)
        Sred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex128)
    
        if ii == 0:
            Hred[:n_state_all,:n_state_all] = np.dot(np.conj(psi_g_1d), HX_g_1d.T )
        
        else:
            for i in range(n_state_all):
                Hred[i,i] = eval0[i]
        
        Hred[:n_state_all,n_state_all:] = np.dot(np.conj(psi_g_1d), HR_1d.T)
        Hred[n_state_all:,n_state_all:] = np.dot(np.conj(R), HR_1d.T)
        Hred[n_state_all:,:n_state_all] = np.conj(Hred[:n_state_all,n_state_all:].T)
        
        Sred[:n_state_all,:n_state_all] = np.diag([1.+ 0.j]*n_state_all)
        Sred[:n_state_all,n_state_all:] = np.dot(np.conj(psi_g_1d), R.T)
        Sred[n_state_all:,n_state_all:] = np.dot(np.conj(R), R.T)
        Sred[n_state_all:,:n_state_all] = np.conj(Sred[:n_state_all,n_state_all:].T)
        

        Hred = (Hred + np.conj(Hred.T)) / 2.
        Sred = (Sred + np.conj(Sred.T)) / 2.
        
        Hred1 = Hermitian(Hred)
        Sred1 = Hermitian(Sred)

        lamda_red, X_red = linalg.eigh(Hred1, Sred1)

        eval1  = lamda_red.real
        
        deval = np.abs(eval1[:n_state_all] - eval0[:n_state_all])
        eval0 = eval1
        
        psi_g_1d = np.dot(X_red[:n_state_all,:n_state_all].T*-1, psi_g_1d) + np.dot(X_red[n_state_all:,:n_state_all].T*-1, R)
        HX_g_1d  = np.dot(X_red[:n_state_all,:n_state_all].T*-1, HX_g_1d)  + np.dot(X_red[n_state_all:,:n_state_all].T*-1, HR_1d)
        
        R =  np.array([lamda_red[i]* psi_g_1d[i] - HX_g_1d[i] for i in range(n_state_all)])
        res = np.array([np.real(np.sum(np.conj(i)*i))**0.5  for i  in R])
        
        if np.mean(deval) <  0.001 or np.sum(res) < 0.0000001:
            break
    # print(HR_1d[:3,0])
    return(eval0[:n_state_all], psi_g_1d[:n_state_all])


def calc_grad_evals( H_psi, psi_g ):

    Ngw     = psi_g.shape[1]
    Nstates = psi_g.shape[0]

    grad = np.zeros((Nstates, Ngw), dtype = np.complex128)

    for i in range(Nstates):
        grad[i] = H_psi[i]
        for j in range(Nstates):
            grad[i] = grad[i] - np.sum(psi_g[j]*H_psi[i]) * psi_g[j]

    return(grad)


def diag_cg(V_loc_r, psi_g_3d, rho_value, beta_nl, g2_vector, g1_vector, \
                  grid_point, gx_vector_mask, n_state_all, ps_list, atom_symbol, vol, state_occupy_list, libxc, eval_threfold=0.01):
    
    Ngw = np.sum(gx_vector_mask)
    d      = np.zeros((n_state_all, Ngw))
    g_old  = np.zeros((n_state_all, Ngw))
    d_old  = np.zeros((n_state_all, Ngw))
    Kg     = np.zeros((n_state_all, Ngw))
    Kg_old = np.zeros((n_state_all, Ngw))
    Xc     = np.zeros((n_state_all, Ngw))
    gt     = np.zeros((n_state_all, Ngw))
    
    V_xc_all = cal_V_xc(rho_value, g1_vector, g2_vector, vol, state_occupy_list, libxc = libxc)
    H_psi = op_H(V_loc_r, V_xc_all, psi_g_3d, rho_value[0], beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
    psi_g_1d = np.array([i[gx_vector_mask] for i in psi_g_3d])
    H_psi_1d = np.array([i[gx_vector_mask] for i in H_psi])
    Hr =  np.dot(np.conj(psi_g_1d), H_psi_1d.T)

    evals, evc = linalg.eigh(Hr)
    Ebands = sum(evals)
    Nstates = len(state_occupy_list)

    evals_old = copy.deepcopy(evals)
    devals    = np.ones(Nstates)    
   
    alpha_t=0.00003
    beta = 0.0
    I_CG_BETA = 2
    for iter0 in range(50):

        H_psi_1d = np.array([i[gx_vector_mask] for i in H_psi])
        g = calc_grad_evals( H_psi_1d, psi_g_1d )
        
        Kg = np.array([i / ( g2_vector[gx_vector_mask] + 1.) for i in g])
    
        if iter0 != 0 :
            if I_CG_BETA == 1:
                beta = np.real(np.sum(np.conj(g)*Kg))/np.real(np.sum(np.conj(g_old)*Kg_old))
            elif I_CG_BETA == 2:
                beta = np.real(np.sum(np.conj(g-g_old)*Kg))/np.real(np.sum(np.conj(g_old)*Kg_old))
            elif I_CG_BETA == 3:
                beta = np.real(np.sum(np.conj(g-g_old)*Kg))/np.real(np.sum(np.conj(g-g_old)*d))
            else:
                beta = np.real(np.sum(np.conj(g)*Kg))/np.real(np.sum((g-g_old)*np.conj(d_old)))
    
        if beta < 0.0:
            beta = 0.0
        
        d = -Kg   + beta * d_old
        tmp_psi = orth1( psi_g_1d + alpha_t*d )
        tmp_psi_g_3d = []
        for i in tmp_psi:
            psi_g_new = np.zeros(grid_point, dtype = np.complex128)
            psi_g_new[gx_vector_mask] = i
            tmp_psi_g_3d.append(psi_g_new)    
        tmp_psi_g_3d = np.array(tmp_psi_g_3d)
        
        
        V_xc_all = cal_V_xc(rho_value, g1_vector, g2_vector, vol, state_occupy_list, libxc = libxc)
        H_psi = op_H(V_loc_r, V_xc_all, tmp_psi_g_3d, rho_value[0], beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
        H_psi_1d = np.array([i[gx_vector_mask] for i in H_psi])
        gt = calc_grad_evals( H_psi_1d, tmp_psi )
        denum = np.real(np.sum(np.conj(g-gt)*d))
        
        if denum != 0.0 :
            alpha = np.abs( alpha_t*np.real(np.sum(np.conj(g)*d))/denum )
        else:
            alpha = 0.0

        psi_g_1d = orth1(psi_g_1d + alpha*d)
        psi_g_3d = []
        for i in psi_g_1d:
            psi_g_new = np.zeros(grid_point, dtype = np.complex128)
            psi_g_new[gx_vector_mask] = i
            psi_g_3d.append(psi_g_new)    
        psi_g_3d = np.array(psi_g_3d)

        rho_r_all = cal_rhoe(psi_g_3d, vol, state_occupy_list, grid_point)
        H_psi = op_H(V_loc_r, V_xc_all, tmp_psi_g_3d, rho_value[0], beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
        H_psi_1d = np.array([i[gx_vector_mask] for i in H_psi])
        Hr =  np.dot(np.conj(psi_g_1d), H_psi_1d.T)
        evals, evc = linalg.eigh(Hr)      
        
        Ebands_old = copy.deepcopy(Ebands)
        Ebands = np.sum(evals)
        devals = np.abs( evals - evals_old )
        evals_old = copy.deepcopy(evals)
        
        nconv = np.sum(devals < 0.001 )
        diffE = np.abs(Ebands-Ebands_old)
        
        g_old = copy.deepcopy(g)
        d_old = copy.deepcopy(d)
        Kg_old = copy.deepcopy(Kg)

                       
        if  nconv  >= Nstates:
            break
               
    return(evals, psi_g_1d)