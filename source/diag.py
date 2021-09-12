import numpy as np
import scipy
from scipy import linalg
from scipy import special


from hamiltionian import *

# 对角化


def diag_davidson(V_loc_r, psi_g_3d, rho_r_all, beta_nl, g2_vector, \
                  grid_point, gx_vector_mask, n_state_all, ps_list, atom_symbol):
    

    HX = op_H(V_loc_r, psi_g_3d, rho_r_all, beta_nl, g2_vector, grid_point, ps_list, atom_symbol)
    HX_g_1d  = np.array([i[gx_vector_mask] for i in HX])
    psi_g_1d = np.array([i[gx_vector_mask] for i in psi_g_3d])

    eval0 = np.array([np.real(np.sum(np.conj(i) * j)) for i, j in zip(psi_g_1d, HX_g_1d) ])
    
    
    R   = np.array([eval0[i] * psi_g_1d[i] - HX_g_1d[i]  for i in range(len(eval0))])
    res = np.array([np.real(np.sum(np.dot(np.conj(i), i))**0.5)  for i in R])
    
    
    Hred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex)
    Sred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex)
    
    for ii in range(50):
        
        res_norm = 1.0 / res
        
        R = np.array([res_norm[i] * R[i] for i in range(n_state_all)])
        
        R = np.array([i / ( g2_vector[gx_vector_mask] + 1.) for i in R])
        
        R_3d  = []
        HR_1d = []
        for i in range(n_state_all):
            tmp_3d = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex)
            tmp_3d[gx_vector_mask] = R[i]
            R_3d.append(tmp_3d)
        
        R_3d = np.array(R_3d)
            
        HR    = op_H(V_loc_r, R_3d, rho_r_all, beta_nl, g2_vector, grid_point, ps_list, atom_symbol)
        HR_1d = np.array([i[gx_vector_mask]  for i in HR])
        
        
        Hred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex)
        Sred = np.zeros((n_state_all*2, n_state_all*2), dtype=np.complex)
    
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
        
        Hred1 = np.triu(Hred) + np.conj(np.triu(Hred)- np.diag(np.diag(Hred))).T  - 1.j*np.diag(np.diag(Hred)).imag
        Sred1 = np.triu(Sred) + np.conj(np.triu(Sred)- np.diag(np.diag(Sred))).T  - 1.j*np.diag(np.diag(Sred)).imag
        

        
        lamda_red, X_red = linalg.eigh(Hred1, Sred1)

        eval1  = lamda_red.real
        

        
        deval = np.abs(eval1[:n_state_all] - eval0[:n_state_all])
        eval0 = eval1
        
       
        psi_g_1d = np.dot(X_red[:n_state_all,:n_state_all].T*-1, psi_g_1d) + np.dot(X_red[n_state_all:,:n_state_all].T*-1, R)
        HX_g_1d  = np.dot(X_red[:n_state_all,:n_state_all].T*-1, HX_g_1d)  + np.dot(X_red[n_state_all:,:n_state_all].T*-1, HR_1d)
        
        R =  np.array([lamda_red[i]* psi_g_1d[i] - HX_g_1d[i] for i in range(n_state_all)])
        res = np.array([np.sum(np.conj(i)* i)**0.5  for i  in R])
        
        # print("% 3d  eval=% 10.5f   deval=% 10.5f  res=% 7.5f"  %(ii+1, eval1[0], np.sum(deval), np.sum(res.real)))

        if np.sum(deval) <  0.0000001:
            break
        
        
    return(eval0[:n_state_all], psi_g_1d[:n_state_all])