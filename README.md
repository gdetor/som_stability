# som_stability

This repository contains all the scripts for reproducing the experiments from 
"Stability analysis of a neural field self-organizing map" ([1]). 

## Contents
- **som.py**  Main simulation script. This script implements the neural field model and the self-organization algorithm. 
- **distortion.py** Implements distortion (using Numba) and MSE functions for measuring the performance of the SOM.
- **som_dxdy.py** Implements the Dx-Dy representation function. It qualifies the topographic map after learning.
- **condition.py** Computes the condition (14) from [1] assuring stability of the SOM model.
- **plot_functions.py** Implemts all the necessary functions for plotting the figures in [1].
- **figureX.py** All scripts named ``figure`` plot the X figure from [1]. The user need to run first the som.py and distortion scripts to get the necessary data files. All the data should be stored in a folder named ``data/``.  

## Requirements
 - Python 3
 - Numpy
 - Numba
 - Sklearn
 - Matlotlib


## Run experiments
In order to run the experiments you can use the script ``som.py`` as follows:
```
$ python som.py 135 0.47 0.2 04702
```
In this case you run the SOM experiment with (K_e, K_i) = (0.47, 0.2) 
and the PRNG seed set to 135.


## Running platform information
The current source code has been tested on the following system configuration:
- CPU Intel i7 10th Generation with 32 GB physical memory
- Linux 5.3.0-29-generic #31-Ubuntu SMP x86_64 x86_64 x86_64 GNU/Linux
- GCC (Ubuntu 9.2.1-9ubuntu2) 9.2.1 20191008
- Python 3.7.5
  - Numpy 1.18.1
  - Numba 0.48.0
  - Sklearn 0.22.1
  - Matlotlib 3.1.3


## License
The current repository is freely available under the GPL 3.0. 


## References
[1] G.Is. Detorakis, A. Chaillet, and N.P. Rougier, 
"Stability analysis of a neural field self-organizing map", 2020.
