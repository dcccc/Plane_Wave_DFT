import numpy as np
import scipy
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
        # res_norm[res_norm < 0.1] = 0.1
        # res_norm[res_norm > 100. ] = 100.


        R = np.array([res_norm[i] * R[i] for i in range(n_state_all)])
        
        R = np.array([i / ( g2_vector[gx_vector_mask] + 1.) for i in R])
        
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
        
        # print("% 3d  eval=% 10.5f   deval=% 10.5f  res=% 7.5f"  %(ii+1, eval1[0], np.sum(deval), np.sum(res.real)))

        psi_g_1d = orth1(psi_g_1d)
        if np.sum(deval) <  0.01 or np.sum(res) < 0.01:
            break
      
    return(eval0[:n_state_all], psi_g_1d[:n_state_all])


def diag_davidson_qe(V_loc_r, psi_g_3d, rho_r_all, beta_nl, g2_vector, g1_vector, \
                  grid_point, gx_vector_mask, N, ps_list, atom_symbol, vol, state_occupy_list):

    HX = op_H(V_loc_r, psi_g_3d, rho_r_all, beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
    HX_g_1d  = np.array([i[gx_vector_mask] for i in HX])
    psi_g_1d = np.array([i[gx_vector_mask] for i in psi_g_3d])
   

    hpsi = np.zeros((N*2, np.sum(gx_vector_mask)), dtype=np.complex128)
    spsi = np.zeros((N*2, np.sum(gx_vector_mask)), dtype=np.complex128)
    psi  = np.zeros((N*2, np.sum(gx_vector_mask)), dtype=np.complex128)

    Hc = np.zeros((N*2, N*2), dtype=np.complex128)
    Sc = np.zeros((N*2, N*2), dtype=np.complex128)
    vc = np.zeros((N*2, N*2), dtype=np.complex128)

    ew = np.zeros(N*2, dtype=np.complex128)

    psi[:N] = psi_g_1d

    hpsi[:N] = HX_g_1d
    spsi[:N] = psi_g_1d



    Hc[:N,:N] = np.dot(np.conj(psi_g_1d), HX_g_1d.T )
    Sc[:N,:N] = np.dot(np.conj(psi_g_1d), psi_g_1d.T )

    ew[:N],  vc[:N, :N]= linalg.eigh(Hc[:N,:N], Sc[:N,:N])
    vc[:N, :N] = vc[:N, :N]*-1.0
    evals = ew[:N].real


    is_conv = np.array([False]*N)

    for ii in range(50):
        num_unconv = N - np.sum(is_conv)
        unconv_idx = np.where(is_conv==False)[0]
        if num_unconv != 0:
            vc[:num_unconv] = vc[unconv_idx]
            ew[N:N+num_unconv] = ew[unconv_idx]
        else:
            break


        psi[N:N+num_unconv] = np.dot(vc[:num_unconv, :N].T, spsi[:N])
        psi[N:N+num_unconv] = -psi[N:N+num_unconv] *ew[N:N+num_unconv].reshape((-1,1))
        psi[N:N+num_unconv] += np.dot(vc[:num_unconv, :N].T, hpsi[:N])

        psi = np.array([i / ( g2_vector[gx_vector_mask] + 1.) for i in psi])
        ew[N:N+num_unconv] = [np.sum(np.conj(i)*i).real for i in psi[N:N+num_unconv]]
        psi[N:N+num_unconv] /=  np.sqrt(ew[num_unconv]).reshape((-1,1))

        tmp_psi_3d = []
        for i in range(N, N+num_unconv):
            tmp_3d = np.zeros((grid_point[0],grid_point[1],grid_point[2]), dtype = np.complex128)
            tmp_3d[gx_vector_mask] = psi[i]
            tmp_psi_3d.append(tmp_3d)
        tmp_psi_3d = np.array(tmp_psi_3d)
        tmp_hpsi = op_H(V_loc_r, tmp_psi_3d, rho_r_all, beta_nl, g2_vector, g1_vector, grid_point, ps_list, atom_symbol, vol, state_occupy_list)
        hpsi[N:N+num_unconv] = np.array([i[gx_vector_mask] for i in tmp_hpsi])
        spsi[N:N+num_unconv] = psi[N:N+num_unconv]


        Hc[N:N+num_unconv,N:N+num_unconv] = np.dot(np.conj(psi[N:N+num_unconv]), hpsi[N:N+num_unconv].T)
        Sc[N:N+num_unconv,N:N+num_unconv] = np.dot(np.conj(psi[N:N+num_unconv]), spsi[N:N+num_unconv].T)

        nbase = N+num_unconv
        for i in range(nbase):
            Hc[i,i] = Hc[i,i].real
            Sc[i,i] = Sc[i,i].real
            for j in range(i, nbase):
                Hc[j,i] = np.conj(Hc[i,j])
                Sc[j,i] = np.conj(Sc[i,j])

        tmp_ew, tmp_vc = linalg.eigh(Hermitian(Hc[:nbase,:nbase]), Hermitian(Sc[:nbase,:nbase]))
        ew[:nbase] = tmp_ew.real
        vc[:nbase, :nbase] = tmp_vc*-1
        vc[N:] = 0.0 + 0.0j

        is_conv = abs(ew[:N] - evals[:N]) < 1.0e-3
        notcnv  = np.sum(is_conv==False)
        evals[:N] = ew[:N]

        if notcnv == 0 or nbase + notcnv > N*2  or ii  > 50:
            psi[:N] = np.dot(vc[:N, :nbase], psi[:nbase])

            if notcnv == 0 or ii > 50:
                break

            
            psi[N:N*2] = np.dot(vc[:N, :nbase], spsi[:nbase] )
            spsi[:N] = psi[N:N*2]
            psi[N:N*2] = np.dot(vc[:N, :nbase], hpsi[:nbase])
            hpsi[:N] = psi[N:N*2]

            nbase = N
            Hc[:nbase,] = 0.0 + 0.0j
            Sc[:nbase,] = 0.0 + 0.0j
            vc[:nbase,] = 0.0 + 0.0j


            for i in range(nbase):
                Hc[i,i] = evals[i]
                Sc[i,i] = 1.0+0.0j
                vc[i,i] = 1.0+0.0j


    return(evals[:N], psi[:N])
