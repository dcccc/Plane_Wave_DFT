import numpy as np
import scipy
from scipy import linalg
from scipy import special



# struct factor

def get_struct_factor(input, g_vector):
    
    atom_symbol = input.atom_symbol
    atom_pos    = input.coord
    grid_point  = input.grid_point
    
    
    g1_vector     = g_vector.g1_vector
    g_vector_mask = g_vector.g_vector_mask
    
    sym_set = set(atom_symbol)

    struct_factor = {}
    for i in set(atom_symbol):
        struct_factor[i] = 0.
    
    for n in range(len(atom_symbol)):
        tmp_sf = np.zeros((grid_point[0], grid_point[1], grid_point[2]), dtype = np.complex)
        tmp_sym = atom_symbol[n]        
        for i in range(grid_point[0]):
            for j in range(grid_point[1]):
                for k in range(grid_point[2]):
                    if g_vector_mask[i, j, k]:
                        tmp_gx = np.sum( atom_pos[n] * g1_vector[i, j, k] )
                        tmp_sf[i, j, k] = np.cos(tmp_gx) - 1.0j * np.sin(tmp_gx)
        struct_factor[tmp_sym] = struct_factor[tmp_sym] + tmp_sf
    return(struct_factor)
