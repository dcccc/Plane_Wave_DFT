

## Plane Wave DFT

A plane-wave basis DFT code, which is written with python. This code is translated from the julia code by f-fathurrahman[1], and the purpose is to get a better understanding about plane-wave basis DFT method.

## How To Run

A sample input file for the calculation of a ch4 molecule in a box is as below

```
e_cut 15
n_state  0

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

4th to 6th line is the box lattice parameter, and unit is Bohr

8th to final line are the atom symbols and corresponding fraction coordinates


When the input file is ready, we can enter into the source directory and run the calculation

```bash
python pwdft.py CH4.txt
```


Then the output text will be printed on the screen.



## Result Compared with CP2k

Total energy of CH4 example calculation is -6.16333698 Hartree, which is same to the result by cp2k. The ch4 input file for cp2k is also included in the example directory.

In the calculation of ch4 example, we use the simplest xalpha[2] functional, and the pseudopotential is norm-conserving potential[3]


## Possible Bugs


1. As to norm-conserving potential, the default cutoff energy of 15.0 Hartree is too less, so the calculation for the example ch4 is not accurate enough

2. Only few sample cases are tested, tasks with elements of larger atom number are not tested, so there may be some wrong results for them

3. The code is pure python and not optimized, so calculation may be unacceptably slow for large systems. You should use a more mature code to do the serious calculation, like quantum espresso

4. As the fixed occupation method used and only the gamma point is considered, the scf will not be converged for delocalized systems. But for localized systems, the scf can converge easily


---------------------------------------------------------------------

## 平面波密度泛函方法

使用平面波基组的密度泛函方法实现代码，代码使用python写成，翻译自f-fathurrahman的Julia[1]实现。主要还在于加深对平面波方法的理解。


## 使用方法

下面是CH4分子的计算输入文件CH4.txt实例

```
e_cut 15
n_state  0

16.0 0.0 0.0
0.0 16.0 0.0
0.0 0.0 16.0

C      0.4570010622     0.5063128759      0.5247731384 
H      0.5859984945     0.5063140570      0.5247731384 
H      0.4140027055     0.3952430396      0.5743205773 
H      0.4140027055     0.5189397899      0.4038105856 
H      0.4140015244     0.6047569793      0.5961894333 

```

第一行设定波函数截断能的大小，这里为15.0 Hatree。 若没有这一行的设定，则默认使用15.0 Hatree。

第二行设定需要计算的态或者轨道的数量，设为0或者没有这一行的设定，以及设定数量小于价体系电子数的一半，则默认使用大于或者等于价电子数量一半的最小整数值

第四行到第六行设定晶格常数，单位为Bohr

第八行开始至末尾设定原子种类和原子的分数坐标

准备好输入文件后，进入source文件夹运行

```bash
python pwdft.py CH4.txt
```

输出便会打印在输出界面上


## 计算结果对比

上面CH4示例代码计算的总能量为-6.16333698 Hartree，cp2k计算的结果为 -6.16333698 Hartree，两者一致。具体输入和结果文件如example文件夹中文件。

计算使用的泛函仍旧是最简单的xalpha泛函[2]，赝势是模守恒赝势[3]


## 可能存在的问题

1. 代码使用的是gth模守恒赝势[3]，默认的15.0 Hatree 较小，因此计算准确度不高

2. 目前仅测试对比了少数几个结果的计算结果，也没有测试原子系数更大的元素体系，因此可能存在一些元素计算结果不对的情况

3. 代码纯粹使用python写成，没有优化速度，计算速度较慢，因此无法计算一些较大的体系。实际的计算应当使用quantum espresso这一类成熟的计算软件

4. 由于仅能使用固定占据方法，不考虑电子自旋的情况下，计算gamma点的能量， 对于离域性稍强的体系计算可能会不收敛，而对于定域性较强的体系可能较容易收敛一些。


## 参考资料


[1]  https://github.com/f-fathurrahman/PWDFT.jl

[2]  J. C. Slater, Phys. Rev. 81, 385 (1951) 

[3]  https://github.com/juerghutter/GTH
