from input_read import *
from psuedo_potential import *
from gvector import *
from structure_factor import *
from wavefunction_density import *
from potential import *
from hamiltionian import *
from diag import *
from scf import *
from cal_E import *

try:
    import pylibxc
    libxc_installed = True
except:
    print("pylibxc is not installed!")
    libxc_installed = False

from sys import argv



print("\n")
print("       ---------------------------      ")
print("       --                       --      ")
print("       --     Plane Wave DFT    --      ")
print("       --       by  dcccc       --      ")
print("       --        2021.9.9       --      ")
print("       --                       --      ")
print("       ---------------------------      ")
print("\n")
print("\n")

print("Read input file")

if len(argv) >=2 :
    
    input_file = argv[1]
    
    input = Input(input_file)
    ps_list = input.ps_list
else:
    print("There is no input file!")
    exit()


print("\n\n")
print("Realspace Lattice Vector ( Bohr )")
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.latt9[0]))
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.latt9[1]))
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.latt9[2]))
print("\n")
print("Reciprocal Lattice Vectors ( Bohr^-1 )")
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.rec_latt[0]))
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.rec_latt[1]))
print("      % 10.6f     % 10.6f     % 10.6f" %tuple(input.rec_latt[2]))

print("\n")
print("Structure Atom Number: %d" %(len(input.coord)))

print("\n") 
print("Atom Coordinate and Valence Electron Number:" )
print("      atom           x               y               z         valence electron num" )
for i,j,k in zip(input.atom_symbol, input.coord, input.atom_charge ):
    print("       %s      % 10.6f      % 10.6f      % 10.6f              %d" %(i, j[0], j[1], j[2], k))

print("\n") 
print("Total Valence Electron Num           : %d " %(sum(input.atom_charge)))
print("Number of occpuied state             : %d " %(input.n_state_occupy))
print("Number of empty state                : %d " %(input.n_state_empty))



g_vector      = G_vector(input.rec_latt, input.wf_cutoff, input.grid_point)
struct_factor = get_struct_factor(input, g_vector)

print("\n") 

print("Eelectron Wavefunction Cutoff Energy : % 6.2f ( Hartree )"  %(input.wf_cutoff))
print("Eelectron Density Cutoff Energy      : % 6.2f ( Hartree )"  %(input.wf_cutoff*4))
# print("\n") 
print("Eelectron Wavefunction Cutoff Energy : % 6.2f ( Hartree )"  %(input.wf_cutoff))
print("Reciprocal Grid Point Number         :   %d    %d    %d"  %(input.grid_point))
print("Wavefunction Point number            :   %d            "  %(np.sum(g_vector.n_gxw)))
print("Reciprocal Point number              :   %d             " %(np.sum(g_vector.n_gw)))
print("\n")
libxc = []
xc_name = input.xc


is_hf = False
exx_alpha = 0.0
# if hartree fock method
if len(xc_name) ==1 and xc_name[0].lower() == "hf":
    libxc = []
    exx_alpha = 1.0
    is_hf = True
# if dft method
elif len(xc_name) > 0 and libxc_installed:
    print("XC functional                        :" )
    for i in xc_name :
        if i.lower() != "hf":
            xc = pylibxc.LibXCFunctional(i, "unpolarized") 
            print(xc.describe())
            libxc.append(xc)
else:
    print("XC functional                        :   Slater X alpha (J. C. Slater, Phys. Rev. 81, 385 (1951) ) " )


# get the local potential and nonlocal potential of psuedo potential
V_loc_r = get_ps_vloc_r(ps_list, g_vector, input, struct_factor) 
beta_nl = get_ps_vnloc_g(input, g_vector, ps_list)


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


gx_vector_mask = g_vector.gx_vector_mask
g1_vector      = g_vector.g1_vector
g2_vector      = g_vector.g2_vector
n_gxw          = g_vector.n_gxw
g_vector_mask  = g_vector.g_vector_mask


# initial wavefunction and density
init_psi_g_3d = get_init_wf(len(state_occupy_list), n_gxw, gx_vector_mask, grid_point)
N_electron = sum(input.atom_charge)
# init_rho_r  =guess_rhoe(struct_factor, g2_vector, gx_vector_mask, atom_symbol, grid_point, ps_list, vol, N_electron  )
init_rho_r = cal_rhoe(init_psi_g_3d,  vol, state_occupy_list, grid_point)

# # when xc functional is MGGA or hybrid functional, using the pbe fuctional results wavefuntion as the initial guess
is_need_tau = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_TAU  for xc in libxc]) 
is_need_lap = np.sum([xc.get_flags() & pylibxc.flags.XC_FLAGS_NEEDS_LAPLACIAN  for xc in libxc])
is_hybrid   = [xc.get_flags() & pylibxc.flags.XC_FAMILY_HYB_LDA  for xc in libxc]
is_hybrid  += [xc.get_flags() & pylibxc.flags.XC_FAMILY_HYB_GGA  for xc in libxc]
is_hybrid  += [xc.get_flags() & pylibxc.flags.XC_FAMILY_HYB_MGGA  for xc in libxc]
is_came_hybrid = [xc.get_flags() & pylibxc.flags.XC_FLAGS_HYB_CAM  for xc in libxc]


# if is_hybrid > 0, get the exx_alpha and cam_parameters, then calculate the coulomb potential for hybrid functional
beta = 0.0
# print(is_hybrid)
if np.sum(is_hybrid) > 0:
    exx_alpha = [xc.get_hyb_exx_coef() for n, xc in enumerate(libxc) 
                 if (is_hybrid[n] or is_hybrid[2*n] or is_hybrid[3*n])and "HYB" == xc_name[n][:3]
                 and not is_came_hybrid[n]]
    # print("exx_alpha list:", exx_alpha)
    exx_alpha = np.sum(exx_alpha)
    cam_parameters =  [xc.get_cam_coef() for n, xc in enumerate(libxc)  if is_came_hybrid[n]]
    if len(cam_parameters) >0:
        exx_alpha += cam_parameters[0][-1]
        beta = cam_parameters[0][0]

if exx_alpha > 0.0:
    print("exx_alpha parameter: % 6.2f" %(exx_alpha))
if beta > 0.0:
    print("CAM hybrid functional, beta parameter: % 6.2f" %(beta))



g_vector.cal_op_coul(beta=beta)
is_hybrid = exx_alpha > 0.0


if is_need_tau > 0 or is_need_lap > 0 or is_hf or is_hybrid > 0:
    print("\n\n")
    print("HF, or xc functionals is MGGA or hybrid functional, using the pbe fuctional results wavefuntion as the initial guess")
    print("Start PBE Self Consistent Field Calculation : ")
    pbe = [pylibxc.LibXCFunctional(i, "unpolarized") for i in ["GGA_X_PBE", "GGA_C_PBE"]]
    eval_final, psi_final, rho_final, E_pw, is_converged = scf(V_loc_r, init_psi_g_3d, init_rho_r, beta_nl, g_vector, input, pbe)
    init_psi_g_3d = psi_final

print("\n\n")
print("Start Self Consistent Field Calculation : ")
print("----------------------------------------------------------------------------------")


# init_psi_g_3d = np.load("/mnt/c/Users/densha/Desktop/qe/Plane_Wave_DFT/example/ch4_hf.save/psi_g_3d.hdf5.npy")
# init_psi_g_3d = np.load("./ch4_hse.save/psi_g_3d.npy")
# init_psi_g_3d = np.load("./pbe_init_psi_g_3d.npy")


eval_final, init_psi_g_3d, init_rho_r, E_pw, is_converged =  scf(V_loc_r, init_psi_g_3d, init_rho_r, beta_nl, \
                          g_vector, input, libxc, 
                          is_hf=is_hf, is_hybrid=is_hybrid, exx_alpha=exx_alpha)


if is_converged :
    print("SCF calculation converageed!")
else:
    print("SCF calculation not converageed after 200 iterations !")
print("\n")    
print("State and Eigenvalue             :   " )  
print("  state num      eigenvalue energy       occupation num")
for n, i, j in zip(np.arange(1,input.n_state_all+1), eval_final, input.state_occupy_list):
    print("    % 3d            %10.6f                %2.1f" %(n,i,j))
    
print("\n")
print("E_kinectic  =  %12.8f (Hatree)"%(E_pw[0]))
print("E_xc        =  %12.8f (Hatree)"%(E_pw[1]))
print("E_hf        =  %12.8f (Hatree)"%(E_pw[2]))
print("E_hatree    =  %12.8f (Hatree)"%(E_pw[3]))
print("E_loc       =  %12.8f (Hatree)"%(E_pw[4]))
print("E_nloc      =  %12.8f (Hatree)"%(E_pw[5]))
print("E_pspcore   =  %12.8f (Hatree)"%(E_pw[6]))
print("E_NN        =  %12.8f (Hatree)"%(E_pw[7]))
print("E_total     =  %12.8f (Hatree)"%(np.sum(E_pw)))