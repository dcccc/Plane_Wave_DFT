import numpy as np
import scipy
from scipy import linalg
from scipy import special


psuedo_potential_dict = {"Ag": "Ag-q11.gth","Ag": "Ag-q19.gth","Al": "Al-q3.gth","Ar": "Ar-q8.gth","As": "As-q5.gth",
                         "At": "At-q7.gth","Au": "Au-q11.gth","Au": "Au-q19.gth","B": "B-q3.gth","Ba": "Ba-q10.gth",
                         "Be": "Be-q4.gth","Bi": "Bi-q15.gth","Bi": "Bi-q5.gth","Br": "Br-q7.gth","C": "C-q4.gth",
                         "Ca": "Ca-q10.gth","Cd": "Cd-q12.gth","Ce": "Ce-q12.gth","Ce": "Ce-q30.gth","Cl": "Cl-q7.gth",
                         "Co": "Co-q17.gth","Cr": "Cr-q14.gth","Cs": "Cs-q9.gth","Cu": "Cu-q11.gth","Cu": "Cu-q19.gth",
                         "Dy": "Dy-q20.gth","Er": "Er-q22.gth","Eu": "Eu-q17.gth","F": "F-q7.gth","Fe": "Fe-q16.gth",
                         "Ga": "Ga-q13.gth","Ga": "Ga-q3.gth","Gd": "Gd-q18.gth","Ge": "Ge-q4.gth","H": "H-q1.gth",
                         "He": "He-q2.gth","Hf": "Hf-q12.gth","Hg": "Hg-q12.gth","Ho": "Ho-q21.gth","I": "I-q7.gth",
                         "In": "In-q13.gth","In": "In-q3.gth","Ir": "Ir-q17.gth","Ir": "Ir-q9.gth","K": "K-q9.gth",
                         "Kr": "Kr-q8.gth","La": "La-q11.gth","Li": "Li-q3.gth","Lu": "Lu-q25.gth","Mg": "Mg-q10.gth",
                         "Mg": "Mg-q2.gth","Mn": "Mn-q15.gth","Mo": "Mo-q14.gth","N": "N-q5.gth","Na": "Na-q9.gth",
                         "Nb": "Nb-q13.gth","Nd": "Nd-q14.gth","Ne": "Ne-q8.gth","Ni": "Ni-q18.gth","O": "O-q6.gth",
                         "Os": "Os-q16.gth","Os": "Os-q8.gth","P": "P-q5.gth","Pb": "Pb-q14.gth","Pb": "Pb-q4.gth",
                         "Pd": "Pd-q10.gth","Pd": "Pd-q18.gth","Pm": "Pm-q15.gth","Po": "Po-q6.gth","Pr": "Pr-q13.gth",
                         "Pt": "Pt-q10.gth","Pt": "Pt-q18.gth","Rb": "Rb-q9.gth","Re": "Re-q15.gth","Re": "Re-q7.gth",
                         "Rh": "Rh-q17.gth","Rh": "Rh-q9.gth","Rn": "Rn-q8.gth","Ru": "Ru-q16.gth","Ru": "Ru-q8.gth",
                         "S": "S-q6.gth","Sb": "Sb-q5.gth","Sc": "Sc-q11.gth","Se": "Se-q6.gth","Si": "Si-q4.gth",
                         "Sm": "Sm-q16.gth","Sn": "Sn-q4.gth","Sr": "Sr-q10.gth","Ta": "Ta-q13.gth","Ta": "Ta-q5.gth",
                         "Tb": "Tb-q19.gth","Tc": "Tc-q15.gth","Te": "Te-q6.gth","Ti": "Ti-q12.gth","Tl": "Tl-q13.gth",
                         "Tl": "Tl-q3.gth","Tm": "Tm-q23.gth","V": "V-q13.gth","W": "W-q14.gth","W": "W-q6.gth",
                         "Xe": "Xe-q8.gth","Y": "Y-q11.gth","Yb": "Yb-q24.gth","Zn": "Zn-q12.gth","Zn": "Zn-q20.gth",
                         "Zr": "Zr-q12.gth" }

						 
						 
						 
						 
# read psuedo potential and input text     
 
class Pspot(object):
    def __init__(self,gth_file):
        self.gth_file = gth_file
        self.gth=open(gth_file,"r").readlines()
        
        self.atom_specie=self.gth[0].split()[0]           # 1
        self.n_spdf=list(map(int,self.gth[1].split()))    # 2
        self.zval=sum(self.n_spdf)
        
        line=self.gth[2].split()                          # 3
        self.r_loc=float(line[0])
        self.n_c_loc=int(line[1])
        self.c_loc=list(map(float,self.gth[3].split()))   # 4
        self.c_loc=self.c_loc + [0.0]*(4-len(self.c_loc))
        
        self.l_max=int(self.gth[4].split()[0])            # 5
        
        n_line=5
        
        self.r_cut=[]
        self.ps_proj=[]
        self.n_proj=[]
        for i in range(self.l_max):
            
            line=self.gth[n_line].split()
            self.r_cut.append(float(line[0]))

            if len(line)>1 and line[1] != "0":
                temp_l=int(line[1])
                temp=np.zeros((temp_l,temp_l))
                self.n_proj.append(temp_l)                  
                for j in range(temp_l):
                    n_line+=1
                    line=self.gth[n_line].split()
                    line=list(map(float,line))
                    temp[j,j:]=np.array(line)
                
                self.ps_proj.append(temp+temp.T-np.diag(np.diag(temp)))
            elif  line[1] == "0":
                self.l_max = self.l_max -1
                
                
            n_line+=1
            
        self.r_cut = self.r_cut + [0.0] * (4-len(self.r_cut))
	

def get_fftw_factor(n):
    
    # good fftw factor n = 2^a * 3^b * 5^c *7 ^d * 11^e * 13^f   and  e/f = 0/1 
    # prime_factor_list = [2, 3, 5, 7, 11, 13]
    # all good fftw factor smaller than 2049
    good_factor_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 
                        27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54, 55, 56, 60, 63, 
                        64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96, 98, 99, 100, 104, 105, 
                        108, 110, 112, 117, 120, 125, 126, 128, 130, 132, 135, 140, 143, 144, 147, 150, 154, 
                        156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 
                        216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260, 264, 270, 273, 275, 
                        280, 286, 288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336, 343, 350, 351, 
                        352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420, 429, 432, 440, 
                        441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525, 528, 539, 
                        540, 546, 550, 560, 567, 572, 576, 585, 588, 594, 600, 616, 624, 625, 630, 637, 640, 
                        648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 715, 720, 728, 729, 735, 750, 756, 
                        768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 858, 864, 875, 880, 882, 891, 
                        896, 900, 910, 924, 936, 945, 960, 972, 975, 980, 990, 1000, 1001, 1008, 1024, 1029, 
                        1040, 1050, 1053, 1056, 1078, 1080, 1092, 1100, 1120, 1125, 1134, 1144, 1152, 1155, 
                        1170, 1176, 1188, 1200, 1215, 1225, 1232, 1248, 1250, 1260, 1274, 1280, 1287, 1296, 
                        1300, 1320, 1323, 1344, 1350, 1365, 1372, 1375, 1386, 1400, 1404, 1408, 1430, 1440, 
                        1456, 1458, 1470, 1485, 1500, 1512, 1536, 1540, 1560, 1568, 1575, 1584, 1600, 1617, 
                        1620, 1625, 1638, 1650, 1664, 1680, 1701, 1715, 1716, 1728, 1750, 1755, 1760, 1764, 
                        1782, 1792, 1800, 1820, 1848, 1872, 1875, 1890, 1911, 1920, 1925, 1944, 1950, 1960, 
                        1980, 2000, 2002, 2016, 2025, 2048]
    
    good_factor_list = np.array(good_factor_list)
    
    assert n < 2048,  print("The grid number %d is too large, can't proceed calculation!" %(n))
    
    delta_n = ( good_factor_list - n ) >= 0
    
    good_n = good_factor_list[delta_n][0]
    
    return(good_n)
	

class Input(object):
    def __init__(self, input_file):

        input_line = open(input_file, "r").readlines()
        
        # read setting parameter for wavefunction cutoff energy 
        for i in range(4):
            line_split = input_line[i].strip().split()
            if len(line_split) == 0:
                n_line = i
                break

            if line_split[0] == "e_cut" :
                self.wf_cutoff   = float(input_line[0].split()[-1])                
            elif line_split[0] != "e_cut" and i > 2:
                print("Wavefunction cutoff not set!")
                print("Use the default value: 15 Hatree!")
                self.wf_cutoff   = 15.
            
            # read state number to compute
            if line_split[0] == "n_state" :
                self.n_state_all = int(input_line[1].split()[-1])                
            elif line_split[0] != "n_state" and i > 2:
                self.n_state_all = 0

            # read xc functional
            if line_split[0] == "xc" :
                self.xc = line_split[1:]
            else:
                self.xc = [] 
                
        
        self.rho_cutoff  =  self.wf_cutoff * 4.
        
        
        # lattice parameter
        if n_line == 0:
            n_line = -1
            
        self.latt9 = np.array([i.split()  for i in input_line[n_line+1:n_line+4]], dtype=np.float64)
        
        self.rec_latt = 2. * np.pi * np.linalg.inv(self.latt9).T
        
        self.vol = np.linalg.det(self.latt9)    

        #read atom fraction coordibate        
        self.frac_coord  = []
        self.atom_symbol = []
        
        for line in input_line[n_line+5:]:
            tmp_line = line.strip().split()
            if len(tmp_line) == 4:
                self.atom_symbol.append(tmp_line[0])
                self.frac_coord.append(tmp_line[1:4])
            elif len(tmp_line) ==0:
                break
        
        self.frac_coord = np.array(self.frac_coord, dtype=np.float64)        
        self.coord  = np.dot(self.frac_coord, self.latt9)
        
        
        # read psuedopotential
        self.ps_list = {}
        
        for i in  set(self.atom_symbol):
            ps_file = "../pseudopotentials/pbe_gth/%s"  %(psuedo_potential_dict[i])
            self.ps_list[i] = Pspot(ps_file)             
        
        self.atom_charge = np.array([ self.ps_list[i].zval for i in self.atom_symbol ])
        
        
            
        # the state number 
        self.n_state_occupy = np.sum(self.atom_charge) / 2.
        
        if self.n_state_all == 0 :
            self.n_state_empty  = 0
            self.n_state_all = self.n_state_occupy
            print("State number not set!")
            print("Use occupied state number!")
        elif self.n_state_occupy >= self.n_state_all :
            self.n_state_empty  = 0
            self.n_state_all = self.n_state_occupy
            print("State number is small than occupied state number!")
            print("Use occupied state number!")
        else:            
            self.n_state_empty  = self.n_state_all - self.n_state_occupy
        
        if self.n_state_occupy % 1 == 0.5:
            self.state_occupy_list = [1.0] * int(self.n_state_occupy) + [0.5] + \
                                     [0.0] * int(self.n_state_empty)
        else:            
            self.state_occupy_list = [1.0] * int(self.n_state_occupy) + \
                                     [0.0] * int(self.n_state_empty)
                
        
        # 计算格点数
        len_latt = np.linalg.norm(self.latt9, axis = 1)
        
        grid_point = 2 * np.round(len_latt * (2 * self.wf_cutoff)**0.5 / np.pi, 0) + 1        
        
        
        self.grid_point=tuple([get_fftw_factor(i) for i in grid_point])
  
  