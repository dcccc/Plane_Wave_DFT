# README.md

Read this in other languages: [中文](README_zh.md).

## Plane Wave DFT

A plane-wave basis DFT code, which is written with python. This code is translated from the julia code by f-fathurrahman[1], and the purpose is to get a better understanding about plane-wave basis DFT method.

## How To Run

A sample input file for the calculation of a ch4 molecule in a box is as below

```
e_cut 15
n_state  0
xc LDA_C_XALPHA 

16.0 0.0 0.0
0.0 16.0 0.0
0.0 0.0 16.0

C      0.4570010622     0.5063128759      0.5247731384 
H      0.5859984945     0.5063140570      0.5247731384 
H      0.4140027055     0.3952430396      0.5743205773 
H      0.4140027055     0.5189397899      0.4038105856 
H      0.4140015244     0.6047569793      0.5961894333 

```

The first line of input text is the wavefunction cutoff energy parameter, the default value is 15.0 Hartree. If this line is omitted, the default value 15 Hartree will be used.

The second line is the number of states you want to calculate. if the parameter is 0, not set, or the value is smaller than the half of the total electron number, an integer of no smaller than half of the total electrons number will be used.

The 3rd line is the functional names in libxc[2] used in the calculation. If not set or pylibxc not installed, the default xalpha[3] will be used.

5th to 7th line is the box lattice parameter, and unit is Bohr

9th to final line are the atom symbols and corresponding fraction coordinates


When the input file is ready, we can enter into the source directory and run the calculation

```bash
python pwdft.py CH4.txt
```


Then the output text will be printed on the screen.



## Result Compared with CP2K and quantum espresso

Total energy of CH4 example calculation is -6.16333698 Hartree, which is same to the result by cp2k. The ch4 input file for cp2k is also included in the example directory. Some other calculations results by quantum espresso is also provided for comparison.

In the calculation of ch4 example, we use the defualt simplest xalpha[3] functional, and the pseudopotential is HGH norm-conserving potential[4]


## Possible Bugs


1. As to norm-conserving potential, the default cutoff energy of 15.0 Hartree is too less, so the calculation for the example ch4 is not accurate enough

2. Only few sample cases are tested, tasks with elements of larger atom number are not tested. So there may be some wrong results for them

3. The code is pure python and not optimized, and calculation may be unacceptably slow for large systems. You should use a more mature code to do the serious calculation, like quantum espresso

4. As the fixed occupation method used here, and only the gamma point is considered. The scf will not be converged for delocalized systems. But for localized systems, the scf can converge easily

## Reference


[1]  https://github.com/f-fathurrahman/PWDFT.jl

[2]  https://gitlab.com/libxc/libxc

[3]  J. C. Slater, Phys. Rev. 81, 385 (1951) 

[4]  https://github.com/juerghutter/GTH, and the converted verison for quantum espresso is from https://pseudopotentials.quantum-espresso.org/legacy_tables/hartwigesen-goedecker-hutter-pp