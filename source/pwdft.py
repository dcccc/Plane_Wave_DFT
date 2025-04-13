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
print("Atom Coordinate and Valence Electron Nummber:" )
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

print("Eelectron Wavefunction Cutoff Energy : % 6.2f ( Hatree )"  %(input.wf_cutoff))
print("Eelectron Density Cutoff Energy      : % 6.2f ( Hatree )"  %(input.wf_cutoff*4))
# print("\n") 
print("Eelectron Wavefunction Cutoff Energy : % 6.2f ( Hatree )"  %(input.wf_cutoff))
print("Reciprocal Grid Point Number         :   %d    %d    %d"  %(input.grid_point))
print("Wavefunction Point number            :   %d            "  %(np.sum(g_vector.n_gxw)))
print("Reciprocal Point number              :   %d             " %(np.sum(g_vector.n_gw)))
print("\n")
libxc = []
xc_name = input.xc
if len(xc_name) > 0 and libxc_installed:
    print("XC functional                        :" )
    for i in xc_name:
        xc = pylibxc.LibXCFunctional(i, "unpolarized")
        print(xc.describe())
        libxc.append(xc)
else:
    print("XC functional                        :   Slater X alpha (J. C. Slater, Phys. Rev. 81, 385 (1951) ) " )



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


# 初始化波函数

init_psi_g_3d = get_init_wf(len(state_occupy_list), n_gxw, gx_vector_mask, grid_point)

init_rho_r  = cal_rhoe(init_psi_g_3d, vol, state_occupy_list, grid_point)

print("\n\n")
print("Start Self Consistent Field Calculation : ")
print("----------------------------------------------------------------------------------")

eval_final, psi_final, rho_final, E_pw, is_converged = scf(V_loc_r, init_psi_g_3d, init_rho_r, beta_nl, g_vector, input, libxc)



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
print("E_hatree    =  %12.8f (Hatree)"%(E_pw[2]))
print("E_loc       =  %12.8f (Hatree)"%(E_pw[3]))
print("E_nloc      =  %12.8f (Hatree)"%(E_pw[4]))
print("E_pspcore   =  %12.8f (Hatree)"%(E_pw[5]))
print("E_NN        =  %12.8f (Hatree)"%(E_pw[6]))
print("E_total     =  %12.8f (Hatree)"%(np.sum(E_pw)))